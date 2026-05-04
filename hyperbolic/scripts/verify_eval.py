"""verify_eval.py — apples-to-apples MPJPE check against MotionAGFormer.

The training loop's `evaluate()` reports `loss_mpjpe(pred, y) * 1000`, which
is the L2 distance in *normalised* coordinates scaled by 1000. That is NOT
the same as the MPJPE that MotionAGFormer / MotionBERT report in mm
camera-space; their pipeline does an additional `denormalize → × 2.5d_factor
→ root-relative` step before computing the error.

This script implements that exact pipeline against your trained checkpoint
so the resulting number is directly comparable to published 38.4 mm and
similar baselines.

Usage
-----
    python scripts/verify_eval.py --run-dir runs/run_YYYYMMDD_HHMMSS
                                  [--ckpt checkpoint_best.pth]
                                  [--source-pkl <abs path to h36m source pkl>]

If --source-pkl is omitted, the script looks for `h36m_sh_conf_cam_source_final.pkl`
(or `h36m_cpn_cam_source.pkl`) one directory above the configured
`data_root`.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network import HyperbolicHPE
from data.reader.h36m import DataReaderH36M
from train import compute_kinematics, generate_topology_matrix


# ─────────────────────────────────────────────────────────────────────────────
# MPJPE helpers (same as MotionAGFormer/MotionBERT)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_mpjpe(pred, gt):
    """pred, gt: [N, J, 3] in mm. Returns per-frame MPJPE in mm: [N]."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def calculate_p_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE. pred, gt: [N, J, 3] in mm. Returns [N]."""
    muX = gt.mean(axis=1, keepdims=True)
    muY = pred.mean(axis=1, keepdims=True)
    X0 = gt - muX
    Y0 = pred - muY
    normX = np.sqrt((X0 ** 2).sum(axis=(1, 2), keepdims=True))
    normY = np.sqrt((Y0 ** 2).sum(axis=(1, 2), keepdims=True))
    X0 /= np.clip(normX, 1e-8, None)
    Y0 /= np.clip(normY, 1e-8, None)
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / np.clip(normY, 1e-8, None)
    t = muX - a * np.matmul(muY, R)
    pred_aligned = a * np.matmul(pred, R) + t
    return np.linalg.norm(pred_aligned - gt, axis=-1).mean(axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", default="checkpoint_best.pth")
    p.add_argument("--config", default=None)
    p.add_argument("--source-pkl", default=None,
                   help="absolute path to the H3.6M source pkl with "
                        "`2.5d_factor`, `joints_2.5d_image`, etc. If omitted, "
                        "script tries to auto-locate next to data_root.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2,
                   help="inference batch size; small default for GPU-contended runs")
    p.add_argument("--max-clips", type=int, default=0,
                   help="if > 0, stop after this many test clips (quick check)")
    return p.parse_args()


def autolocate_source_pkl(data_root):
    """Search the parent of data_root for a recognisable H3.6M source pkl."""
    parent = Path(data_root).resolve()
    candidates = [
        parent / "h36m_sh_conf_cam_source_final.pkl",
        parent / "h36m_cpn_cam_source.pkl",
        parent.parent / "h36m_sh_conf_cam_source_final.pkl",
        parent.parent / "h36m_cpn_cam_source.pkl",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def build_model(cfg, device):
    temporal_windows = (cfg.get("temporal_windows", None)
                        if hasattr(cfg, "get")
                        else getattr(cfg, "temporal_windows", None))
    return HyperbolicHPE(
        in_features=3,
        embed_dim=cfg.embed_dim,
        num_spatial=cfg.num_spatial,
        num_temporal=cfg.num_temporal,
        num_heads=int(getattr(cfg, "num_heads", 8)),
        temporal_window=cfg.temporal_window,
        temporal_windows=temporal_windows,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        num_joints=cfg.num_joints,
    ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    opts = parse_args()
    device = torch.device(opts.device)

    # ── Config + checkpoint ──────────────────────────────────────────────────
    run_dir = Path(opts.run_dir)
    cfg_path = Path(opts.config) if opts.config else (run_dir / "config.yaml")
    print(f"Config: {cfg_path}")
    with open(cfg_path) as f:
        cfg = edict(yaml.safe_load(f))

    ckpt_path = run_dir / "checkpoints" / opts.ckpt
    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  epoch       = {ckpt.get('epoch', '?')}")
    print(f"  best_mpjpe  = {ckpt.get('best_mpjpe', '?')}  "
          f"(train.py units, ≈ normalised loss)")

    # ── Source pkl ───────────────────────────────────────────────────────────
    src_pkl = opts.source_pkl or autolocate_source_pkl(cfg.data_root)
    if not src_pkl or not Path(src_pkl).exists():
        sys.exit(
            "Could not locate the H3.6M source pkl.\n"
            f"Searched near data_root={cfg.data_root}\n"
            "Pass --source-pkl <abs path to h36m_*_cam_source*.pkl>."
        )
    print(f"Source pkl: {src_pkl}")
    src_dir, src_name = os.path.split(src_pkl)

    # ── DataReaderH36M (matches MotionAGFormer's eval exactly) ───────────────
    n_frames = int(cfg.n_frames)
    datareader = DataReaderH36M(
        n_frames=n_frames,
        sample_stride=1,
        data_stride_train=n_frames // 3,
        data_stride_test=n_frames,
        dt_root=src_dir,
        dt_file=src_name,
    )

    print("Reading + slicing test data...")
    train_data, test_data, train_labels, test_labels = datareader.get_sliced_data()
    print(f"  test_data  shape: {test_data.shape}     (clips, T, J, 3)")
    print(f"  test_labels shape: {test_labels.shape}  (clips, T, J, 3)")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}\n")

    # ── Inference (batched) ──────────────────────────────────────────────────
    topo = generate_topology_matrix(cfg.num_joints, device)
    n_clips_total = test_data.shape[0]
    n_clips_eval = n_clips_total
    if opts.max_clips and opts.max_clips < n_clips_total:
        n_clips_eval = int(opts.max_clips)
        print(f"  --max-clips: evaluating first {n_clips_eval} of "
              f"{n_clips_total} clips (others left as zeros, skipped downstream)")
    bs = int(opts.batch_size)
    # Keep results_all full-shape so datareader.denormalize() — which indexes
    # test_hw by absolute clip id — does not mis-align. Unevaluated rows stay
    # at zero and are explicitly skipped in the per-action accumulation below.
    results_all = np.zeros_like(test_labels)
    n_clips = n_clips_eval  # used for inference loop only

    import time
    print(f"Running inference: {n_clips} clips, batch size {bs}, on {device}...")
    print(f"  (live ticks every batch; '.'=ok, 's'=>5s/batch, 'S'=>15s/batch)")
    t_start = time.time()
    last_log_t = t_start
    n_batches = (n_clips + bs - 1) // bs
    with torch.no_grad():
        for bi, i in enumerate(range(0, n_clips, bs)):
            j = min(i + bs, n_clips)
            tb0 = time.time()
            x = torch.from_numpy(test_data[i:j]).float().to(device)
            x_vel = compute_kinematics(x)
            pred = model(x, x_vel, topo)
            if device.type == "cuda":
                torch.cuda.synchronize()
            results_all[i:j] = pred.cpu().numpy()
            dt = time.time() - tb0
            tick = "." if dt < 5 else ("s" if dt < 15 else "S")
            sys.stdout.write(tick)
            sys.stdout.flush()
            # Periodic ETA line
            if time.time() - last_log_t > 30 or bi == n_batches - 1:
                elapsed = time.time() - t_start
                rate = (bi + 1) / max(1e-9, elapsed)
                remaining = (n_batches - bi - 1) / max(1e-9, rate)
                sys.stdout.write(
                    f"\n  [{bi+1}/{n_batches} batches | "
                    f"clips {j}/{n_clips} | "
                    f"{elapsed:.1f}s elapsed | "
                    f"{rate:.2f} batch/s | "
                    f"ETA {remaining:.0f}s]\n"
                )
                sys.stdout.flush()
                last_log_t = time.time()
    print("\nDone.\n")

    # ── Replicate MotionAGFormer's evaluate() exactly ────────────────────────
    print("Applying MotionAGFormer eval pipeline:")
    print("  1) datareader.denormalize(pred)")
    print("  2) pred *= 2.5d_factor")
    print("  3) root-relative subtraction")
    print("  4) MPJPE in mm  (per-action mean → mean across actions)")
    print()

    # 1) Denormalize predictions back to image-pixel-equivalent space
    results_all = datareader.denormalize(results_all)   # → [N_clips, T, J, 3]

    # 2) Pull factors and ground-truth in 2.5d image form
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts     = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    gt_clips     = gts[split_id_test]
    num_test_frames = len(actions)

    # MotionAGFormer skips these three known-corrupt sequences
    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']

    # 3+4) Per-frame errors, accumulated then averaged per action
    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc     = np.zeros(num_test_frames)
    frame_clips = np.array(range(num_test_frames))[split_id_test]

    skipped_partial = 0
    for idx in range(len(action_clips)):
        # Skip clips beyond what we actually ran inference on (matters with
        # --max-clips; results_all rows past n_clips_eval are uninitialised
        # zeros and would corrupt the metric if included).
        if idx >= n_clips_eval:
            skipped_partial += 1
            continue
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx] * factor                # ← KEY denorm step

        # Root-relative
        pred = pred - pred[:, 0:1, :]
        gt   = gt   - gt[:, 0:1, :]

        err1 = calculate_mpjpe(pred, gt)
        err2 = calculate_p_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1

    action_names = sorted(set(actions))
    results, results_p = {a: [] for a in action_names}, {a: [] for a in action_names}
    for idx in range(num_test_frames):
        if oc[idx] > 0:
            results[actions[idx]].append(e1_all[idx] / oc[idx])
            results_p[actions[idx]].append(e2_all[idx] / oc[idx])

    per_action = []
    per_action_p = []
    for a in action_names:
        if results[a]:
            per_action.append(np.mean(results[a]))
            per_action_p.append(np.mean(results_p[a]))
    e1 = float(np.mean(per_action))
    e2 = float(np.mean(per_action_p))

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print("RESULTS  (MotionAGFormer eval pipeline)")
    print("=" * 72)
    if skipped_partial > 0:
        coverage = 100.0 * (1 - skipped_partial / len(action_clips))
        print(f"  ⚠ partial eval: {coverage:.1f}% of test clips "
              f"({skipped_partial}/{len(action_clips)} skipped via --max-clips)")
    print(f"  Protocol #1 MPJPE    = {e1:7.2f}  mm       ← compare to MotionAGFormer 38.4")
    print(f"  Protocol #2 P-MPJPE  = {e2:7.2f}  mm")
    print()
    print("Per-action MPJPE (mm):")
    for a, v in zip(action_names, per_action):
        print(f"  {a:<14s} {v:6.2f}")
    print()
    print("=" * 72)
    print("CONTRAST with train.py reporting")
    print("=" * 72)
    print(f"  train.py best_mpjpe (raw)        = {ckpt.get('best_mpjpe', '?')}")
    print(f"  train.py 'mm' (× 1000)            = {float(ckpt.get('best_mpjpe', 0.0)) * 1000:.2f}")
    print(f"  Proper MPJPE (mm camera-space)   = {e1:.2f}")
    ratio = e1 / max(1e-9, float(ckpt.get('best_mpjpe', 1e-9)) * 1000)
    print(f"  Ratio (proper / train.py-shown)  = {ratio:.3f}")
    print()
    if abs(ratio - 1.0) < 0.05:
        print("  → train.py 'mm' is approximately correct (within 5%).")
    else:
        print(f"  → train.py 'mm' is OFF by ~{abs(ratio - 1) * 100:.0f}%.")
        print(f"     The headline number for the paper should be {e1:.2f} mm, "
              f"NOT {float(ckpt.get('best_mpjpe', 0)) * 1000:.2f} mm.")


if __name__ == "__main__":
    main()
