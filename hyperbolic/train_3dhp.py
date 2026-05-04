"""
train_3dhp.py — MPI-INF-3DHP training for HyperbolicHPE v3 (hpe_SOTA).
Mirrors the logging/checkpointing style of train.py.
Extra eval metrics: PCK@150mm and AUC (standard 3DHP benchmarks).
"""

import os, sys, argparse, yaml, csv, json, shutil, logging, time, socket, platform
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, k):
            if k in self: return self[k]
            raise AttributeError(k)
        def __setattr__(self, k, v): self[k] = v

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network import HyperbolicHPE
from data.reader.motion_dataset import MPI3DHP
from utils.learning import (
    decay_lr_exponentially, build_cosine_warmup_scheduler, AverageMeter,
)
from utils.tools import set_random_seed
from loss.pose3d import loss_mpjpe
from loss.hyperbolic_loss import (
    geodesic_velocity_loss, geodesic_bone_loss,
    manifold_drift_loss, UncertaintyWeightedLoss,
)

# ── Mirror items ──────────────────────────────────────────────────────────────
_MIRROR_ITEMS = ["train_3dhp.py", "configs", "model", "math_utils", "loss",
                 "data/reader", "utils"]


# ── RunManager ────────────────────────────────────────────────────────────────
class RunManager:
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

    @classmethod
    def new(cls, base="runs_3dhp"): return cls(Path(base) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    @classmethod
    def resume_from(cls, run_dir):
        rm = cls(Path(run_dir)); rm.log(f"Resuming from {run_dir}"); return rm

    def _setup_logger(self):
        self.logger = logging.getLogger(f"hpe.3dhp.{id(self)}")
        self.logger.setLevel(logging.DEBUG); self.logger.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.log_txt, mode="a", encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
        self.logger.addHandler(fh); self.logger.addHandler(ch)

    def log(self, msg, level="info"): getattr(self.logger, level)(msg)

    def append_csv(self, row):
        needs_header = not self.log_csv.exists() or not self._csv_ready
        with open(self.log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if needs_header: w.writeheader(); self._csv_ready = True
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

    def mirror_source(self):
        for item in _MIRROR_ITEMS:
            src = Path(PROJECT_ROOT) / item
            if not src.exists(): continue
            dst = self.src_dir / item; dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists(): shutil.rmtree(dst)
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else: shutil.copy2(src, dst)
        self.log(f"Source mirrored → {self.src_dir}")

    def save_config(self, args, config_path):
        shutil.copy2(config_path, self.run_dir / "config.yaml")
        with open(self.run_dir / "config_resolved.json", "w") as f: json.dump(args, f, indent=2, default=str)

    def save_env_info(self):
        info = {"timestamp": datetime.now().isoformat(), "hostname": socket.gethostname(),
                "platform": platform.platform(), "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}
        with open(self.run_dir / "env_info.json", "w") as f: json.dump(info, f, indent=2)
        self.log(f"GPU: {info['gpu_name']} | PyTorch {info['torch']}")

    def _ckpt_path(self, tag):
        if tag == "latest": return self.ckpt_dir / self.CKPT_LATEST
        if tag == "best":   return self.ckpt_dir / self.CKPT_BEST
        return self.ckpt_dir / f"checkpoint_{tag}.pth"

    def save_checkpoint(self, state, tag):
        p = self._ckpt_path(tag); tmp = p.with_suffix(".tmp")
        torch.save(state, tmp); tmp.replace(p)

    def load_checkpoint(self, tag="latest", device="cpu"):
        p = self._ckpt_path(tag)
        if not p.exists(): raise FileNotFoundError(f"Checkpoint not found: {p}")
        ckpt = torch.load(p, map_location=device, weights_only=False)
        self.log(f"Loaded {p.name}  (epoch {ckpt.get('epoch','?')})"); return ckpt

    def build_state(self, epoch, model, optimizer, loss_weighter, lr, best_mpjpe, args, metrics=None):
        return {"epoch": epoch, "best_mpjpe": best_mpjpe, "lr": lr,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "config": dict(args), "metrics": metrics or {},
                "timestamp": datetime.now().isoformat()}


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    type=str, default="configs/hyperbolic_hpe_3dhp.yaml")
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--pretrained",type=str, default=None, help="Path to H36M pretrained checkpoint")
    p.add_argument("--runs-dir",  type=str, default="runs_3dhp")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


def load_pretrained_weights(model, ckpt):
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    md = model.state_dict()
    matched, discarded = [], []
    new_sd = OrderedDict()
    for k, v in sd.items():
        k = k[7:] if k.startswith("module.") else k
        if k in md and v.shape == md[k].shape:
            new_sd[k] = v; matched.append(k)
        else:
            discarded.append(k)
    md.update(new_sd); model.load_state_dict(md)
    return len(matched), len(discarded)


def compute_kinematics(x):
    v = torch.zeros_like(x); T = x.shape[1]
    if T > 2:
        v[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5
        v[:, 0] = x[:, 1] - x[:, 0]; v[:, -1] = x[:, -1] - x[:, -2]
    elif T > 1:
        v[:, 1:] = x[:, 1:] - x[:, :-1]; v[:, 0] = v[:, 1]
    return v


def generate_topology_matrix(num_joints, device):
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    topo = torch.zeros((num_joints, num_joints), device=device)
    for child, parent in enumerate(parents):
        if parent >= 0: topo[child, parent] = topo[parent, child] = 1.0
    return topo


def riemannian_loss_curriculum_weight(epoch, warmup_start=10, warmup_end=20):
    if epoch < warmup_start: return 0.0
    if epoch >= warmup_end:  return 1.0
    return (epoch - warmup_start) / max(1, warmup_end - warmup_start)


def augment_joint_confidence_dropout(x, p_sample=0.2, max_drop=2):
    if not x.requires_grad: x = x.clone()
    B, T, J, _ = x.shape
    mask = torch.rand(B, device=x.device) < p_sample
    if not mask.any(): return x
    for b in mask.nonzero(as_tuple=False).flatten().tolist():
        n = int(torch.randint(1, max_drop + 1, (1,)).item())
        x[b, :, torch.randperm(J, device=x.device)[:n], 2] = 0.0
    return x


def _fmt(v, prec=4): return f"{v:.{prec}f}"


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(opts, args, model, loss_weighter, dataloader,
                    optimizer, device, epoch, rm, epoch_pbar, scaler, lr_scheduler=None):
    model.train()
    meters = {k: AverageMeter() for k in ["total", "mpjpe", "vel", "bone", "drift"]}
    topo_bias = generate_topology_matrix(args.num_joints, device)
    nan_steps = 0
    curric_w = riemannian_loss_curriculum_weight(
        epoch,
        warmup_start=int(getattr(args, "loss_curriculum_start", 10)),
        warmup_end=int(getattr(args, "loss_curriculum_end", 20)),
    )
    aug_p_drop = float(getattr(args, "joint_conf_dropout_p", 0.2))
    eval_scale = float(getattr(args, "eval_scale", 1024.0))

    step_pbar = tqdm(dataloader, desc=f"  Ep {epoch:03d} [train]",
                     unit="batch", leave=False, dynamic_ncols=True, colour="cyan")

    for step, (x, y) in enumerate(step_pbar):
        B = x.shape[0]
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        if aug_p_drop > 0.0:
            x = augment_joint_confidence_dropout(x, p_sample=aug_p_drop)

        x_vel = compute_kinematics(x)

        if args.root_rel:
            y = y - y[..., 14:15, :]   # joint 14 = pelvis/root in 3DHP

        optimizer.zero_grad()

        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred, h_manifold = model(x, x_vel, topo_bias, return_manifold=True)

            l_mpjpe = loss_mpjpe(pred, y)

            # Normalize to unit-scale before geodesic losses to avoid scale explosion
            # (3DHP coords can be ~1024x larger than H36M normalized space)
            scale = pred.detach().abs().max().clamp(min=1e-6)
            pred_n, y_n = pred / scale, y / scale
            l_vel  = (geodesic_velocity_loss(pred_n, y_n)).clamp(max=10.0)
            l_bone = (geodesic_bone_loss(pred_n, y_n)).clamp(max=10.0)
            
            if getattr(args, "disable_lvel", False):
                l_vel = torch.tensor(0.0, device=device)
            if getattr(args, "disable_lbone", False):
                l_bone = torch.tensor(0.0, device=device)

            l_drift = manifold_drift_loss(h_manifold.view(-1, h_manifold.shape[-1]))

            if getattr(args, "fixed_loss_weights", False):
                loss_total = l_mpjpe + curric_w * l_vel + curric_w * l_bone + l_drift
                weights = [1.0, curric_w, curric_w, 1.0]
            else:
                loss_total, weights = loss_weighter(
                    l_mpjpe, curric_w * l_vel, curric_w * l_bone, l_drift)

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            nan_steps += 1; optimizer.zero_grad(); continue

        scaler.scale(loss_total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_weighter.parameters()),
            max_norm=float(getattr(args, "grad_clip", 1.0)))
        scaler.step(optimizer); scaler.update()
        if lr_scheduler is not None: lr_scheduler.step()

        meters["total"].update(loss_total.item(), B)
        meters["mpjpe"].update(l_mpjpe.item(), B)
        meters["vel"].update(l_vel.item(), B)
        meters["bone"].update(l_bone.item(), B)
        meters["drift"].update(l_drift.item(), B)

        drift_val = meters["drift"].avg
        step_pbar.set_postfix(OrderedDict(
            loss=_fmt(meters["total"].avg),
            mpjpe_mm=_fmt(meters["mpjpe"].avg * eval_scale, 1),
            log_drift=_fmt(torch.log10(torch.tensor(drift_val + 1e-9)).item(), 2),
        ))

    step_pbar.close()
    if nan_steps: rm.log(f"  ↳ {nan_steps} NaN/Inf steps skipped", level="warning")
    return meters


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(opts, args, model, dataloader, device, epoch, rm):
    model.eval()
    topo_bias = generate_topology_matrix(args.num_joints, device)
    eval_scale   = float(getattr(args, "eval_scale", 1024.0))
    pck_thr      = float(getattr(args, "pck_threshold", 150.0))
    mpjpe_m = AverageMeter(); pck_m = AverageMeter(); auc_m = AverageMeter()

    eval_pbar = tqdm(dataloader, desc=f"  Ep {epoch:03d} [eval] ",
                     unit="batch", leave=False, dynamic_ncols=True, colour="yellow")

    with torch.no_grad():
        for batch in eval_pbar:
            # Support both (x, y_norm) and (x, y_norm, y_mm, valid, seq) formats
            if len(batch) == 2:
                x, y_norm = batch
                y_mm = None
            else:
                x, y_norm, y_mm, *_ = batch

            B = x.shape[0]
            x, y_norm = x.to(device), y_norm.to(device)
            x_vel = compute_kinematics(x)
            pred = model(x, x_vel, topo_bias)

            if args.root_rel:
                pred   = pred   - pred[..., 14:15, :]
                y_norm = y_norm - y_norm[..., 14:15, :]

            mpjpe_m.update(loss_mpjpe(pred, y_norm).item(), B)

            # PCK and AUC: work in mm
            pred_mm = pred * eval_scale
            if y_mm is not None:
                gt_mm = y_mm.to(device)
            else:
                gt_mm = y_norm * eval_scale
            gt_mm = gt_mm - gt_mm[..., 14:15, :]

            dist_mm = torch.norm(pred_mm - gt_mm, dim=-1)   # [B, T, J]
            pck = (dist_mm < pck_thr).float().mean() * 100.0
            pck_m.update(pck.item(), B)

            thrs = np.arange(0, pck_thr + 5, 5)
            auc = np.mean([(dist_mm < t).float().mean().item() for t in thrs]) * 100.0
            auc_m.update(auc, B)

            eval_pbar.set_postfix(
                mpjpe_mm=f"{mpjpe_m.avg * eval_scale:.1f}",
                pck=f"{pck_m.avg:.1f}%")

    eval_pbar.close()
    return {"mpjpe": mpjpe_m.avg, "pck": pck_m.avg, "auc": auc_m.avg}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Run setup ─────────────────────────────────────────────────────────────
    if opts.resume:
        rm = RunManager.resume_from(opts.resume)
        ckpt = rm.load_checkpoint("latest", device=device)
        args = edict(ckpt["config"])
        start_epoch = ckpt["epoch"] + 1
        best_mpjpe  = ckpt["best_mpjpe"]
        lr          = ckpt["lr"]
        rm.log(f"Resuming from epoch {start_epoch} | best_mpjpe={best_mpjpe:.4f}")
    else:
        rm = RunManager.new(opts.runs_dir)
        with open(opts.config) as f: args = edict(yaml.safe_load(f))
        start_epoch = 0; best_mpjpe = float("inf"); lr = args.learning_rate; ckpt = None
        rm.save_config(dict(args), opts.config); rm.save_env_info(); rm.mirror_source()

    rm.log(f"Run directory : {rm.run_dir}")
    rm.log(f"Device        : {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    temporal_windows = args.get("temporal_windows", None) if hasattr(args, "get") \
        else getattr(args, "temporal_windows", None)
    model = HyperbolicHPE(
        in_features=3, embed_dim=args.embed_dim,
        num_spatial=args.num_spatial, num_temporal=args.num_temporal,
        num_heads=getattr(args, "num_heads", 8),
        temporal_window=args.temporal_window,
        temporal_windows=temporal_windows,
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        num_joints=args.num_joints,
    ).to(device)

    loss_weighter = UncertaintyWeightedLoss(4).to(device)
    optimizer = AdamW(
        list(model.parameters()) + list(loss_weighter.parameters()),
        lr=lr, weight_decay=args.weight_decay)

    # ── Load pretrained H36M weights (optional) ───────────────────────────────
    if opts.pretrained and ckpt is None:
        pt = torch.load(opts.pretrained, map_location="cpu", weights_only=False)
        m, d = load_pretrained_weights(model, pt)
        rm.log(f"Pretrained weights loaded: {m} matched, {d} discarded")

    # ── Restore checkpoint states ─────────────────────────────────────────────
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        for pg in optimizer.param_groups: pg["lr"] = lr
        rm.log("Model / optimizer / loss-weighter states restored.")

    n_params = sum(p.numel() for p in model.parameters())
    rm.log(f"HyperbolicHPE v3 | Parameters: {n_params:,}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = MPI3DHP(args, train=True)
    test_dataset  = MPI3DHP(args, train=False)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=4, pin_memory=True)
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
    rm.log(f"Training on {len(train_dataset)} samples | Testing on {len(test_dataset)} samples")

    # ── AMP + LR scheduler ────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler('cuda')
    if ckpt is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    steps_per_epoch = len(train_loader)
    warmup_epochs   = int(getattr(args, "warmup_epochs", 5))
    lr_scheduler = build_cosine_warmup_scheduler(
        optimizer,
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=warmup_epochs * steps_per_epoch,
        min_lr_ratio=float(getattr(args, "min_lr_ratio", 0.01)),
    )
    if ckpt is not None and "lr_scheduler_state_dict" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    elif ckpt is not None and start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch): lr_scheduler.step()

    rm.log(f"Cosine+warmup LR | warmup={warmup_epochs} epochs | total={args.epochs} epochs")
    rm.log(f"Batch={args.batch_size} | LR={lr:.2e} | n_frames={args.n_frames} | stride={getattr(args,'stride',81)}")

    eval_scale = float(getattr(args, "eval_scale", 1024.0))

    # ── Epoch loop ────────────────────────────────────────────────────────────
    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training", unit="ep",
                      initial=start_epoch, total=args.epochs, dynamic_ncols=True, colour="green")

    for epoch in epoch_pbar:
        t0 = time.time()

        meters = train_one_epoch(
            opts, args, model, loss_weighter, train_loader,
            optimizer, device, epoch, rm, epoch_pbar,
            scaler=scaler, lr_scheduler=lr_scheduler)

        eval_metrics = evaluate(opts, args, model, test_loader, device, epoch, rm)

        elapsed   = time.time() - t0
        eval_mpjpe = eval_metrics["mpjpe"]
        is_best   = eval_mpjpe < best_mpjpe
        if is_best: best_mpjpe = eval_mpjpe
        lr = optimizer.param_groups[0]["lr"]

        # ── Epoch summary (matches train.py style + PCK/AUC) ──────────────────
        epoch_pbar.set_postfix(OrderedDict(
            tr_mm=_fmt(meters["mpjpe"].avg * eval_scale, 1),
            ev_mm=_fmt(eval_mpjpe * eval_scale, 1),
            pck=f"{eval_metrics['pck']:.1f}",
            auc=f"{eval_metrics['auc']:.1f}",
            best_mm=_fmt(best_mpjpe * eval_scale, 1),
            lr=f"{lr:.2e}", t=f"{elapsed:.0f}s",
        ))

        rm.log(
            f"Epoch {epoch:03d}/{args.epochs-1} | "
            f"Loss {meters['total'].avg:.4f} | "
            f"TrainMPJPE {meters['mpjpe'].avg * eval_scale:.1f}mm | "
            f"EvalMPJPE {eval_mpjpe * eval_scale:.1f}mm | "
            f"PCK {eval_metrics['pck']:.1f}% | "
            f"AUC {eval_metrics['auc']:.1f}% | "
            f"Drift(log10) {torch.log10(torch.tensor(meters['drift'].avg + 1e-9)).item():.2f} | "
            f"Best {best_mpjpe * eval_scale:.1f}mm | "
            f"LR {lr:.2e} | {elapsed:.1f}s"
            + (" ★ NEW BEST" if is_best else "")
        )

        rm.append_csv({
            "epoch": epoch,
            "train_loss": meters["total"].avg,
            "train_mpjpe_mm": meters["mpjpe"].avg * eval_scale,
            "eval_mpjpe_mm": eval_mpjpe * eval_scale,
            "eval_pck": eval_metrics["pck"],
            "eval_auc": eval_metrics["auc"],
            "best_mpjpe_mm": best_mpjpe * eval_scale,
            "lr": lr, "elapsed_sec": elapsed, "is_best": int(is_best),
        })

        # ── Checkpoints ───────────────────────────────────────────────────────
        state = rm.build_state(epoch, model, optimizer, loss_weighter, lr, best_mpjpe, args,
                               metrics={"train_mpjpe_mm": meters["mpjpe"].avg * eval_scale,
                                        "eval_mpjpe_mm": eval_mpjpe * eval_scale,
                                        "eval_pck": eval_metrics["pck"],
                                        "eval_auc": eval_metrics["auc"]})
        state["scaler_state_dict"] = scaler.state_dict()
        state["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

        rm.save_checkpoint(state, "latest")
        if is_best:
            rm.save_checkpoint(state, "best")
            rm.log(f"  ↳ Saved best (eval_mpjpe={best_mpjpe * eval_scale:.1f}mm | PCK={eval_metrics['pck']:.1f}%)")
        if (epoch + 1) % 10 == 0:
            rm.save_checkpoint(state, f"epoch_{epoch+1:04d}")
            rm.log(f"  ↳ Saved interval checkpoint: epoch_{epoch+1:04d}")

    epoch_pbar.close()
    rm.log(f"Training complete. Best EvalMPJPE: {best_mpjpe * eval_scale:.1f}mm")
    rm.log(f"Run artifacts: {rm.run_dir}")


if __name__ == "__main__":
    main()
