"""
train.py — Research-grade training for HyperbolicHPE v2.

Features
--------
  ▸ Immutable run directories  (runs/run_YYYYMMDD_HHMMSS/)
  ▸ Source mirror              (src/ copy of all model/loss/utils files)
  ▸ Config + env-info snapshot (config.yaml, config_resolved.json, env_info.json)
  ▸ Structured CSV log         (train_log.csv — one row per epoch)
  ▸ Rotating text log          (train.log — full DEBUG stream)
  ▸ Checkpoints                (latest / best / every-10-epoch interval)
  ▸ Complete resumability      (--resume <run_dir>)
  ▸ Rich tqdm display          (nested bars with live metric postfix)
  ▸ Optional W&B               (--use-wandb)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os, sys, argparse, yaml, csv, json, shutil, logging, time, socket, platform
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from easydict import EasyDict as edict

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# ── Project ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network import HyperbolicHPE
from model.lorentz_network import LorentzHPE
from data.reader.motion_dataset import MotionDataset3D
from data.reader.h36m import DataReaderH36M
from utils.learning import (
    decay_lr_exponentially, build_cosine_warmup_scheduler, AverageMeter,
)
from utils.tools import set_random_seed
from loss.pose3d import loss_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.hyperbolic_loss import (
    geodesic_velocity_loss, geodesic_bone_loss,
    manifold_drift_loss, UncertaintyWeightedLoss,
)
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# RunManager
# ─────────────────────────────────────────────────────────────────────────────
_MIRROR_ITEMS = [
    "train.py", "configs",
    "model", "math_utils", "loss",
    "data/reader", "utils",
]


class RunManager:
    """Handles all run-level I/O: directories, mirror, logs, checkpoints."""

    CKPT_LATEST = "checkpoint_latest.pth"
    CKPT_BEST   = "checkpoint_best.pth"

    def __init__(self, run_dir: Path):
        self.run_dir  = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.src_dir  = self.run_dir / "src"
        self.log_csv  = self.run_dir / "train_log.csv"
        self.log_txt  = self.run_dir / "train.log"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.src_dir.mkdir(exist_ok=True)

        self._csv_ready = False
        self._setup_logger()

    # ── Constructors ──────────────────────────────────────────────────────────
    @classmethod
    def new(cls, base: str = "runs") -> "RunManager":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(Path(base) / f"run_{stamp}")

    @classmethod
    def resume_from(cls, run_dir: str) -> "RunManager":
        rm = cls(Path(run_dir))
        rm.log(f"Resuming from {run_dir}")
        return rm

    # ── Logger ────────────────────────────────────────────────────────────────
    def _setup_logger(self):
        self.logger = logging.getLogger(f"hpe.{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.log_txt, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log(self, msg: str, level: str = "info"):
        getattr(self.logger, level)(msg)

    # ── CSV log ───────────────────────────────────────────────────────────────
    def append_csv(self, row: dict):
        needs_header = not self.log_csv.exists() or not self._csv_ready
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if needs_header:
                writer.writeheader()
                self._csv_ready = True
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})

    # ── Source mirror ─────────────────────────────────────────────────────────
    def mirror_source(self):
        for item in _MIRROR_ITEMS:
            src = Path(PROJECT_ROOT) / item
            if not src.exists():
                continue
            dst = self.src_dir / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst,
                                ignore=shutil.ignore_patterns(
                                    "__pycache__", "*.pyc", ".pytest_cache"))
            else:
                shutil.copy2(src, dst)
        self.log(f"Source mirrored → {self.src_dir}")

    # ── Config + env ─────────────────────────────────────────────────────────
    def save_config(self, args: dict, config_path: str):
        shutil.copy2(config_path, self.run_dir / "config.yaml")
        with open(self.run_dir / "config_resolved.json", "w") as f:
            json.dump(args, f, indent=2, default=str)
        self.log(f"Config saved → {self.run_dir / 'config.yaml'}")

    def save_env_info(self):
        info = {
            "timestamp":     datetime.now().isoformat(),
            "hostname":      socket.gethostname(),
            "platform":      platform.platform(),
            "python":        sys.version,
            "torch":         torch.__version__,
            "cuda_version":  torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name":      (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else "N/A"),
            "gpu_count":     torch.cuda.device_count(),
        }
        try:
            import subprocess
            info["git_hash"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            info["git_hash"] = "unavailable"
        with open(self.run_dir / "env_info.json", "w") as f:
            json.dump(info, f, indent=2)
        self.log(f"GPU: {info['gpu_name']} | "
                 f"PyTorch {info['torch']} | CUDA {info['cuda_version']}")

    # ── Checkpoints ───────────────────────────────────────────────────────────
    def _ckpt_path(self, tag: str) -> Path:
        if tag == "latest":
            return self.ckpt_dir / self.CKPT_LATEST
        if tag == "best":
            return self.ckpt_dir / self.CKPT_BEST
        return self.ckpt_dir / f"checkpoint_{tag}.pth"

    def save_checkpoint(self, state: dict, tag: str):
        path = self._ckpt_path(tag)
        tmp  = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)   # atomic on Windows+Linux; rename() fails on Windows if dst exists

    def load_checkpoint(self, tag: str = "latest", device="cpu") -> dict:
        path = self._ckpt_path(tag)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.log(f"Loaded {path.name}  (epoch {ckpt.get('epoch', '?')})")
        return ckpt

    def build_state(self, epoch, model, optimizer, loss_weighter,
                    lr, best_mpjpe, args, metrics=None) -> dict:
        return {
            "epoch":                    epoch,
            "best_mpjpe":               best_mpjpe,
            "lr":                       lr,
            "model_state_dict":         model.state_dict(),
            "optimizer_state_dict":     optimizer.state_dict(),
            "loss_weighter_state_dict": loss_weighter.state_dict(),
            "config":                   dict(args),
            "metrics":                  metrics or {},
            "timestamp":                datetime.now().isoformat(),
            "torch_version":            str(torch.__version__),
            "run_dir":                  str(self.run_dir),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HyperbolicHPE v2 — Research Training")
    p.add_argument("--config",    type=str, default="configs/hyperbolic_hpe.yaml",
                   help="Path to YAML config (ignored when --resume is set)")
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to a previous run directory to resume from")
    p.add_argument("--runs-dir",  type=str, default="runs",
                   help="Root directory for new run folders")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


def compute_kinematics(x):
    """Central-difference finite-difference velocities. x: [B, T, J, 3]

    Central differences (v_t = (x_{t+1} - x_{t-1}) / 2) have ~2× the SNR of
    backward differences for the same noise level on x, and are unbiased to
    second order — preferred over the previous backward-difference scheme.
    Boundary frames fall back to forward / backward differences.
    """
    v = torch.zeros_like(x)
    T = x.shape[1]
    if T > 2:
        v[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5
        v[:, 0]    = x[:, 1] - x[:, 0]
        v[:, -1]   = x[:, -1] - x[:, -2]
    elif T > 1:
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        v[:, 0]  = v[:, 1]
    return v


def generate_topology_matrix(num_joints, device):
    """H36M kinematic-tree adjacency matrix."""
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    topo = torch.zeros((num_joints, num_joints), device=device)
    for child, parent in enumerate(parents):
        if parent >= 0:
            topo[child, parent] = 1.0
            topo[parent, child] = 1.0
    return topo


def _fmt(v, prec=4):
    return f"{v:.{prec}f}"


def augment_joint_confidence_dropout(x, p_sample=0.2, max_drop=2):
    """Randomly zero out the confidence channel of a few joints per sample.

    Synergises with the confidence-gated embedding: simulating CPN failures
    in training teaches the network to route around low-confidence joints
    via the topology bias and temporal context. Only the conf channel
    (x[..., 2]) is touched; coordinates are untouched, so the 3D label is
    unaffected.

    x: [B, T, J, 3]   training-time only.
    """
    if not x.requires_grad:
        x = x.clone()
    B, T, J, _ = x.shape
    # Per-sample mask: which samples in the batch get joint-dropout this step.
    sample_mask = torch.rand(B, device=x.device) < p_sample
    if not sample_mask.any():
        return x
    # For each affected sample, pick 1..max_drop random joints to zero.
    for b in sample_mask.nonzero(as_tuple=False).flatten().tolist():
        n_drop = int(torch.randint(1, max_drop + 1, (1,)).item())
        idx = torch.randperm(J, device=x.device)[:n_drop]
        x[b, :, idx, 2] = 0.0
    return x


def riemannian_loss_curriculum_weight(epoch, warmup_start=10, warmup_end=20):
    """Ramp weight α ∈ [0, 1] for the geodesic velocity / bone losses.

    They measure manifold geometry of *predictions* — at the start of training
    predictions are essentially random, so these terms inject pure noise into
    the gradient. Zero them out until the network has learned something
    meaningful, then ramp linearly to full weight.

    Default schedule: 0.0 for the first 10 epochs, linear 0→1 over epochs
    10–19, 1.0 from epoch 20 onwards. Kendall uncertainty weighting then
    handles the steady-state magnitude balance from there.
    """
    if epoch < warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return 1.0
    return (epoch - warmup_start) / max(1, warmup_end - warmup_start)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(opts, args, model, loss_weighter, dataloader,
                    optimizer, device, epoch, rm: RunManager,
                    epoch_pbar, scaler, lr_scheduler=None):
    model.train()
    meters = {k: AverageMeter()
              for k in ["total", "mpjpe", "vel", "bone", "drift"]}
    topo_bias = generate_topology_matrix(args.num_joints, device)
    nan_steps = 0

    # Curriculum weight for L_vel and L_bone (0 early, 1 after warm-up).
    curric_w = riemannian_loss_curriculum_weight(
        epoch,
        warmup_start=int(getattr(args, "loss_curriculum_start", 10)),
        warmup_end=int(getattr(args, "loss_curriculum_end", 20)),
    )

    step_pbar = tqdm(dataloader,
                     desc=f"  Ep {epoch:03d} [train]",
                     unit="batch", leave=False,
                     dynamic_ncols=True, colour="cyan")

    aug_p_drop = float(getattr(args, "joint_conf_dropout_p", 0.2))

    for step, (x, y) in enumerate(step_pbar):
        B, T, J, C = x.shape
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Train-time augmentation: confidence dropout on input keypoints.
        # Done before x_vel so the velocity branch sees the augmented stream
        # too — but velocity drops Δconf inside the embedding, so this is
        # really only a position-side perturbation.
        if aug_p_drop > 0.0:
            x = augment_joint_confidence_dropout(x, p_sample=aug_p_drop)

        x_vel = compute_kinematics(x)

        if args.root_rel:
            y = y - y[..., 0:1, :]

        optimizer.zero_grad()

        # AMP forward + loss under bfloat16
        # bfloat16 chosen over float16: same dynamic range as fp32 →
        # safe for acosh/cosh/sinh in Lorentz ops, no NaN from underflow.
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, h_manifold = model(x, x_vel, topo_bias, return_manifold=True)

            l_mpjpe = loss_mpjpe(pred, y)
            l_vel   = geodesic_velocity_loss(pred, y)
            l_bone  = geodesic_bone_loss(pred, y)
            l_drift = manifold_drift_loss(h_manifold.view(-1, h_manifold.shape[-1]))
            # Curriculum scaling on the Riemannian losses — see
            # riemannian_loss_curriculum_weight.
            loss_total, weights = loss_weighter(
                l_mpjpe, curric_w * l_vel, curric_w * l_bone, l_drift,
            )

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            nan_steps += 1
            rm.log(f"[WARN] NaN/Inf at epoch {epoch} step {step} — skipped",
                   level="warning")
            optimizer.zero_grad()
            continue

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)      # unscale before clip so norms are correct
        # max_norm=1.0: with the Tier-1 EPS/clamp fixes the gradient scale is
        # well-behaved (smoke shows ~1e-3 max norm), so the previous 0.5 was
        # cutting legitimate signal. 1.0 is the standard transformer default.
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_weighter.parameters()),
            max_norm=float(getattr(args, "grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()

        meters["total"].update(loss_total.item(), B)
        meters["mpjpe"].update(l_mpjpe.item(), B)
        meters["vel"].update(l_vel.item(),   B)
        meters["bone"].update(l_bone.item(), B)
        meters["drift"].update(l_drift.item(), B)

        # Live step postfix — MPJPE×1000 = mm, drift as log10 for readability
        drift_val = meters["drift"].avg
        step_pbar.set_postfix(OrderedDict(
            loss    = _fmt(meters["total"].avg),
            mpjpe_mm= _fmt(meters["mpjpe"].avg * 1000, 1),
            vel     = _fmt(meters["vel"].avg),
            bone    = _fmt(meters["bone"].avg),
            log_drift=_fmt(torch.log10(torch.tensor(drift_val + 1e-9)).item(), 2),
        ))

        # ── W&B step log ─────────────────────────────────────────────────────
        if opts.use_wandb:
            wd = {
                "train_step/loss_total":        loss_total.item(),
                "train_step/loss_mpjpe":        l_mpjpe.item(),
                "train_step/loss_vel":          l_vel.item(),
                "train_step/loss_bone":         l_bone.item(),
                "train_step/manifold_drift":    l_drift.item(),
                "train_step/weight_mpjpe":      weights[0],
                "train_step/weight_vel":        weights[1],
                "train_step/weight_bone":       weights[2],
                "epoch": epoch + step / len(dataloader),
            }
            for i, blk in enumerate(model.spatial_blocks):
                wd[f"tau/spatial_{i}_mean"] = blk.attn.tau.mean().item()
            for i, blk in enumerate(model.temporal_blocks):
                wd[f"tau/temporal_{i}_mean"] = blk.attn.tau.mean().item()
            wandb.log(wd)

    step_pbar.close()
    if nan_steps:
        rm.log(f"  ↳ {nan_steps} NaN/Inf steps skipped this epoch", level="warning")

    return meters


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — True-mm MotionAGFormer protocol
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(opts, args, model, dataloader, datareader, device, epoch, rm: RunManager):
    """Evaluate using the standard H3.6M protocol:
       denormalize → apply 2.5d_factor → root-relative → MPJPE/P-MPJPE in mm.
    """
    model.eval()
    topo_bias = generate_topology_matrix(args.num_joints, device)

    eval_pbar = tqdm(dataloader,
                     desc=f"  Ep {epoch:03d} [eval] ",
                     unit="batch", leave=False,
                     dynamic_ncols=True, colour="yellow")

    # Collect all predictions in normalized space
    results_all = []
    with torch.no_grad():
        for x, y in eval_pbar:
            x = x.to(device, non_blocking=True)
            x_vel = compute_kinematics(x)
            pred  = model(x, x_vel, topo_bias)

            if args.root_rel:
                pred[:, :, 0, :] = 0  # Zero root (same convention as MotionAGFormer)

            results_all.append(pred.cpu().numpy())

    eval_pbar.close()
    results_all = np.concatenate(results_all)

    # Denormalize predictions (reverse /res_w * 2)
    results_all = datareader.denormalize(results_all)

    # Load GT metadata from the raw pickle
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts     = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips  = frames[split_id_test]
    gt_clips     = gts[split_id_test]

    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc     = np.zeros(num_test_frames)

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']

    results_by_action = {}
    results_p_by_action = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for a in action_names:
        results_by_action[a] = []
        results_p_by_action[a] = []

    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt   = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        pred = pred - pred[:, 0:1, :]
        gt   = gt   - gt[:, 0:1, :]

        err1 = calculate_mpjpe(pred, gt)
        err2 = calculate_p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1

    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            action = actions[idx]
            results_by_action[action].append(e1_all[idx] / oc[idx])
            results_p_by_action[action].append(e2_all[idx] / oc[idx])

    e1 = np.mean([np.mean(results_by_action[a]) for a in action_names])
    e2 = np.mean([np.mean(results_p_by_action[a]) for a in action_names])

    if opts.use_wandb:
        wandb.log({"eval_epoch/mpjpe_mm": e1,
                   "eval_epoch/p_mpjpe_mm": e2,
                   "epoch": epoch + 1})

    return e1, e2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Run manager setup ────────────────────────────────────────────────────
    if opts.resume:
        rm = RunManager.resume_from(opts.resume)
        ckpt = rm.load_checkpoint("latest", device=device)
        args = edict(ckpt["config"])
        args.lr = args.learning_rate
        start_epoch = ckpt["epoch"] + 1
        best_mpjpe  = ckpt["best_mpjpe"]
        lr          = ckpt["lr"]
        rm.log(f"Resuming from epoch {start_epoch} | best_mpjpe={best_mpjpe:.4f}")
    else:
        rm = RunManager.new(opts.runs_dir)
        with open(opts.config, "r") as f:
            args = edict(yaml.safe_load(f))
        args.lr = args.learning_rate
        start_epoch = 0
        best_mpjpe  = float("inf")
        lr          = args.lr
        ckpt        = None
        rm.save_config(dict(args), opts.config)
        rm.save_env_info()
        rm.mirror_source()

    rm.log(f"Run directory : {rm.run_dir}")
    rm.log(f"Device        : {device}")

    # ── W&B ─────────────────────────────────────────────────────────────────
    if opts.use_wandb:
        wandb.init(
            project="Hyperbolic_PhaseSpace_HPE",
            name=rm.run_dir.name,
            dir=str(rm.run_dir),
            resume="allow" if opts.resume else None,
            config=dict(args),
        )

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = MotionDataset3D(args, args.subset_list, "train")
    test_dataset  = MotionDataset3D(args, args.subset_list, "test")
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0,
                               pin_memory=True)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size,
                               shuffle=False, num_workers=0,
                               pin_memory=True)

    # ── DataReaderH36M for proper mm-scale evaluation ────────────────────────
    dt_file = getattr(args, 'dt_file', 'h36m_sh_conf_cam_source_final.pkl')
    datareader = DataReaderH36M(
        n_frames=args.n_frames, sample_stride=1,
        data_stride_train=args.n_frames // 3,
        data_stride_test=args.n_frames,
        dt_root=args.data_root, dt_file=dt_file,
    )
    rm.log(f"DataReaderH36M loaded (dt_file={dt_file}) for true-mm evaluation")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_version = getattr(args, 'model_version', 'v3')
    if model_version == 'v4':
        model = LorentzHPE(
            in_features = 3,
            embed_dim   = args.embed_dim,
            n_layers    = getattr(args, 'n_layers', 12),
            num_heads   = getattr(args, 'num_heads', 8),
            mlp_ratio   = args.mlp_ratio,
            dropout     = args.dropout,
            num_joints  = args.num_joints,
        ).to(device)
    else:
        temporal_windows = args.get("temporal_windows", None) if hasattr(args, "get") \
            else getattr(args, "temporal_windows", None)
        model = HyperbolicHPE(
            in_features      = 3,
            embed_dim        = args.embed_dim,
            num_spatial      = args.num_spatial,
            num_temporal     = args.num_temporal,
            num_heads        = getattr(args, "num_heads", 8),
            temporal_window  = args.temporal_window,
            temporal_windows = temporal_windows,
            mlp_ratio        = args.mlp_ratio,
            dropout          = args.dropout,
            num_joints       = args.num_joints,
        ).to(device)

    loss_weighter = UncertaintyWeightedLoss(4).to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(loss_weighter.parameters()),
        lr=lr, weight_decay=args.weight_decay,
    )

    # ── Restore states if resuming ────────────────────────────────────────────
    if ckpt is not None:
        model.state_dict()          # warm-up
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        rm.log("Model / optimizer / loss-weighter states restored.")

    n_params = sum(p.numel() for p in model.parameters())
    rm.log(f"{'LorentzHPE v4' if model_version == 'v4' else 'HyperbolicHPE v3'} | Parameters: {n_params:,}")
    rm.log(f"Epochs: {start_epoch} → {args.epochs - 1} | "
           f"LR: {lr:.2e} | Batch: {args.batch_size}")

    # ── AMP GradScaler ───────────────────────────────────────────────────────
    # bfloat16: same dynamic range as fp32, no underflow risk on Lorentz ops.
    # GradScaler is kept for compatibility but is a no-op with bfloat16;
    # it becomes critical if dtype is switched to float16.
    scaler = torch.amp.GradScaler('cuda')
    if ckpt is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        rm.log("GradScaler state restored.")
    rm.log(f"AMP: bfloat16  |  GradScaler: enabled")

    # ── LR scheduler: linear warmup → cosine decay (per-step) ────────────────
    # Replaces the previous per-epoch exponential decay (γ=0.99). Cosine with
    # 5-epoch warmup is the standard for transformer pose lifters; the warmup
    # avoids early-step gradient instability when the joint embedding and
    # confidence gate are still random, and cosine gives a smoother end-of-
    # training profile than exponential.
    steps_per_epoch = len(train_loader)
    warmup_epochs = int(getattr(args, "warmup_epochs", 5))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_scheduler = build_cosine_warmup_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
        min_lr_ratio=float(getattr(args, "min_lr_ratio", 0.01)),
    )
    if ckpt is not None and 'lr_scheduler_state_dict' in ckpt:
        lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        rm.log("LR scheduler state restored.")
    # Fast-forward scheduler if resuming from a non-zero epoch but no saved
    # scheduler state (older checkpoints). Step it `start_epoch * steps_per_epoch`
    # times so the LR profile matches the resumed position.
    elif ckpt is not None and start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            lr_scheduler.step()
        rm.log(f"LR scheduler fast-forwarded {start_epoch * steps_per_epoch} steps.")
    rm.log(f"LR schedule  : cosine + linear warmup "
           f"({warmup_epochs} warmup epochs, {args.epochs} total)")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    epoch_pbar = tqdm(range(start_epoch, args.epochs),
                      desc="Training", unit="ep",
                      initial=start_epoch, total=args.epochs,
                      dynamic_ncols=True, colour="green")

    for epoch in epoch_pbar:
        t0 = time.time()

        # Train
        meters = train_one_epoch(
            opts, args, model, loss_weighter,
            train_loader, optimizer, device, epoch, rm, epoch_pbar,
            scaler=scaler, lr_scheduler=lr_scheduler,
        )

        # Evaluate (true mm via MotionAGFormer protocol)
        eval_mpjpe, eval_p_mpjpe = evaluate(opts, args, model, test_loader,
                                            datareader, device, epoch, rm)

        elapsed = time.time() - t0
        is_best = eval_mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = eval_mpjpe

        # Read current LR for logging — scheduler updates per-step inside
        # train_one_epoch so we just observe wherever it landed.
        lr = optimizer.param_groups[0]["lr"]

        # ── Epoch postfix (outer bar) — true mm from MotionAGFormer protocol ─
        epoch_pbar.set_postfix(OrderedDict(
            ev_mm   = _fmt(eval_mpjpe, 1),
            p_mm    = _fmt(eval_p_mpjpe, 1),
            best_mm = _fmt(best_mpjpe, 1),
            lr      = f"{lr:.2e}",
            t       = f"{elapsed:.0f}s",
        ))

        # ── Text log — true mm ────────────────────────────────────────────────
        rm.log(
            f"Epoch {epoch:03d}/{args.epochs-1} | "
            f"Loss {meters['total'].avg:.4f} | "
            f"Vel {meters['vel'].avg:.4f} | "
            f"Bone {meters['bone'].avg:.4f} | "
            f"Drift(log10) {torch.log10(torch.tensor(meters['drift'].avg+1e-9)).item():.2f} | "
            f"Eval {eval_mpjpe:.1f}mm | "
            f"P-MPJPE {eval_p_mpjpe:.1f}mm | "
            f"Best {best_mpjpe:.1f}mm | "
            f"LR {lr:.2e} | "
            f"{elapsed:.1f}s"
            + (" \u2605 NEW BEST" if is_best else "")
        )

        # ── CSV log — true mm ─────────────────────────────────────────────────
        rm.append_csv({
            "epoch":              epoch,
            "train_loss_total":   meters["total"].avg,
            "train_vel":          meters["vel"].avg,
            "train_bone":         meters["bone"].avg,
            "train_log10_drift":  torch.log10(torch.tensor(meters["drift"].avg + 1e-9)).item(),
            "eval_mpjpe_mm":      eval_mpjpe,
            "eval_p_mpjpe_mm":    eval_p_mpjpe,
            "best_mpjpe_mm":      best_mpjpe,
            "lr":                 lr,
            "elapsed_sec":        elapsed,
            "is_best":            int(is_best),
        })

        # ── Checkpoints ──────────────────────────────────────────────────────────
        state = rm.build_state(
            epoch, model, optimizer, loss_weighter, lr, best_mpjpe, args,
            metrics={
                "eval_mpjpe_mm":   eval_mpjpe,
                "eval_p_mpjpe_mm": eval_p_mpjpe,
            },
        )
        state["scaler_state_dict"] = scaler.state_dict()  # for AMP resume
        state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

        rm.save_checkpoint(state, "latest")          # always

        if is_best:
            rm.save_checkpoint(state, "best")        # new best eval
            rm.log(f"  ↳ Saved best checkpoint (eval_mpjpe={best_mpjpe:.4f})")

        if (epoch + 1) % 10 == 0:                    # interval snapshot
            tag = f"epoch_{epoch+1:04d}"
            rm.save_checkpoint(state, tag)
            rm.log(f"  ↳ Saved interval checkpoint: {tag}")

        # ── W&B epoch log ─────────────────────────────────────────────────────
        if opts.use_wandb:
            wandb.log({
                "epoch/train_loss":    meters["total"].avg,
                "epoch/eval_mpjpe_mm": eval_mpjpe,
                "epoch/eval_p_mpjpe_mm": eval_p_mpjpe,
                "epoch/best_mpjpe_mm": best_mpjpe,
                "epoch/lr":            lr,
                "epoch/elapsed_sec":   elapsed,
                "epoch": epoch,
            })

    # ── Done ──────────────────────────────────────────────────────────────────
    epoch_pbar.close()
    rm.log(f"Training complete. Best eval MPJPE: {best_mpjpe:.1f} mm")
    rm.log(f"Run artifacts at: {rm.run_dir}")

    if opts.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
