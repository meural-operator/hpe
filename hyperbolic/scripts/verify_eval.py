"""verify_eval.py — independent sanity check on the MPJPE units that
train.py reports.

What this does
--------------
Loads a saved checkpoint, runs the test loader once, and reports the eval
metric three ways so you can confirm whether the headline number is in
real millimetres of camera-space MPJPE or in some other unit:

  (A)  RAW MEAN: mean Euclidean distance between pred and target in
       *whatever units the dataloader stores* (this is what loss_mpjpe
       computes inside train.evaluate).
  (B)  CONVENTION: (A) × 1000 — the value train.py displays as "mm".
  (C)  RANGE PROBE: descriptive stats on the target tensor itself
       (per-channel range, std, root-relative norms) so you can decide
       whether the storage unit is metres, mm, normalised image-pixel,
       or normalised "meta" units.

Then it interprets the result:

  * If target values lie roughly in [-1, +1] but represent METRES
    (the MotionAGFormer / MotionBERT convention for H3.6M cam-source
    pkls), then (B) is the real MPJPE in mm and the train.py output
    is correct.

  * If targets lie in [-1, +1] and represent normalised IMAGE-PIXEL
    coordinates (e.g. raw `joint3d_image / res_w * 2`), the conversion
    factor depends on per-camera intrinsics and (B) is *not* directly
    comparable to other papers' MPJPE.

Usage
-----
    python scripts/verify_eval.py --run-dir runs/run_YYYYMMDD_HHMMSS
                                  [--ckpt checkpoint_best.pth]
                                  [--max-batches 50]    # quick check

The script does NOT require any training to be running; it is a
read-only consumer of the checkpoint and the same test loader train.py
uses, so it produces directly comparable numbers.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network import HyperbolicHPE
from data.reader.motion_dataset import MotionDataset3D
from loss.pose3d import loss_mpjpe
from train import compute_kinematics, generate_topology_matrix


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="path to a runs/run_YYYYMMDD_HHMMSS directory")
    p.add_argument("--ckpt", default="checkpoint_best.pth",
                   help="checkpoint file inside run-dir/checkpoints/")
    p.add_argument("--config", default=None,
                   help="override config; default = run-dir/config.yaml")
    p.add_argument("--max-batches", type=int, default=0,
                   help="if > 0, stop after this many test batches (quick smoke)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def build_model(cfg, device):
    """Mirror the constructor used in train.py."""
    temporal_windows = (cfg.get("temporal_windows", None)
                        if hasattr(cfg, "get")
                        else getattr(cfg, "temporal_windows", None))
    model = HyperbolicHPE(
        in_features      = 3,
        embed_dim        = cfg.embed_dim,
        num_spatial      = cfg.num_spatial,
        num_temporal     = cfg.num_temporal,
        num_heads        = int(getattr(cfg, "num_heads", 8)),
        temporal_window  = cfg.temporal_window,
        temporal_windows = temporal_windows,
        mlp_ratio        = cfg.mlp_ratio,
        dropout          = cfg.dropout,
        num_joints       = cfg.num_joints,
    ).to(device)
    return model


def describe_tensor(name, t):
    """Per-channel stats on a 3-channel pose tensor [..., 3]."""
    flat = t.detach().reshape(-1, t.shape[-1]).float().cpu().numpy()
    mins = flat.min(axis=0); maxs = flat.max(axis=0); stds = flat.std(axis=0)
    print(f"  {name:<22} per-channel min={np.round(mins, 3).tolist()}, "
          f"max={np.round(maxs, 3).tolist()}, std={np.round(stds, 3).tolist()}")


def interpret_units(y_stats):
    """Heuristic: classify the storage unit of the GT joints."""
    abs_max = max(abs(v) for v in y_stats["max"]) \
              if y_stats["max"] is not None else 0.0
    if abs_max <= 2.5:
        return ("normalised", "values in roughly [-1, 1]; either metres "
                              "(MotionBERT/MotionAGFormer convention — "
                              "× 1000 IS millimetres) or image-pixel "
                              "normalisation (× 1000 is NOT mm)")
    if 50 < abs_max < 1500:
        return ("pixels", "values in [-500, +500] range; image-pixel space, "
                          "needs deprojection to get mm")
    if abs_max >= 200:
        return ("millimetres", "values in [-1000, +1000] mm range; raw "
                               "camera-space, loss IS mm directly (no × 1000 "
                               "needed)")
    return ("unknown", "could not classify automatically; inspect the per-channel "
                       "stats above")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    opts = parse_args()
    device = torch.device(opts.device)

    # ── Load config ──────────────────────────────────────────────────────────
    run_dir = Path(opts.run_dir)
    cfg_path = Path(opts.config) if opts.config else (run_dir / "config.yaml")
    if not cfg_path.exists():
        # train.py also saves config_resolved.json — but we want the YAML form
        # used at construction. Fall back to a sibling config.
        cfg_path = run_dir / "config.yaml"
    print(f"Loading config from: {cfg_path}")
    with open(cfg_path) as f:
        cfg = edict(yaml.safe_load(f))
    cfg.lr = cfg.learning_rate

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = run_dir / "checkpoints" / opts.ckpt
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"  checkpoint epoch       = {ckpt.get('epoch', '?')}")
    print(f"  checkpoint best_mpjpe  = {ckpt.get('best_mpjpe', '?')}  "
          f"(this is what train.py stored; same unit as below)")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Test loader (mirror train.py) ────────────────────────────────────────
    print("\nBuilding test dataset (this can take a moment on first run)...")
    test_dataset = MotionDataset3D(cfg, cfg.subset_list, "test")
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    n_clips = len(test_dataset)
    print(f"Test dataset: {n_clips} clips of T={cfg.n_frames} frames each\n")

    topo = generate_topology_matrix(cfg.num_joints, device)

    # ── Probe the first batch's value ranges ─────────────────────────────────
    first_x, first_y = next(iter(test_loader))
    first_x, first_y = first_x.to(device), first_y.to(device)
    first_y_rr = first_y - first_y[..., 0:1, :]   # root-relative

    print("=" * 72)
    print("VALUE-RANGE PROBE (first batch)")
    print("=" * 72)
    describe_tensor("x (input 2D+conf)", first_x)
    describe_tensor("y (raw target 3D)", first_y)
    describe_tensor("y root-relative", first_y_rr)

    y_stats = {
        "min": [float(first_y_rr[..., c].min()) for c in range(3)],
        "max": [float(first_y_rr[..., c].max()) for c in range(3)],
        "std": [float(first_y_rr[..., c].std()) for c in range(3)],
    }
    unit_class, unit_help = interpret_units(y_stats)
    print(f"\n  → Inferred storage unit: \033[1m{unit_class}\033[0m")
    print(f"     {unit_help}")
    print()

    # ── Sweep the full test set computing all variants ───────────────────────
    print("=" * 72)
    print("MPJPE SWEEP (test set)")
    print("=" * 72)
    sum_loss = 0.0    # (A) raw L2 distance in storage units, summed weight = num samples
    sum_n = 0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            B = x.shape[0]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x_vel = compute_kinematics(x)

            pred = model(x, x_vel, topo)
            if cfg.root_rel:
                pred = pred - pred[..., 0:1, :]
                y    = y    - y[..., 0:1, :]

            loss = loss_mpjpe(pred, y).item()    # mean L2 over all (B, T, J)
            sum_loss += loss * B
            sum_n += B
            n_batches += 1
            if opts.max_batches and n_batches >= opts.max_batches:
                print(f"  (stopped early after {n_batches} batches per --max-batches)")
                break

    raw_mpjpe = sum_loss / max(1, sum_n)
    convention_mpjpe = raw_mpjpe * 1000.0

    print(f"  Batches processed       : {n_batches}")
    print(f"  Clips processed         : {sum_n} / {n_clips}")
    print()
    print(f"  (A) RAW mean L2 distance        = {raw_mpjpe:.6f}  "
          f"[storage units]")
    print(f"  (B) train.py CONVENTION × 1000  = {convention_mpjpe:.2f}     "
          f"[interpreted as mm by train.py]")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if unit_class == "normalised":
        # Could be metres or normalised pixels. Both store data in [-1, 1]-ish
        # range, so we can't disambiguate from value range alone. The
        # MotionBERT/MotionAGFormer h36m cam-source pkls store METRES root-
        # relative, in which case (B) IS millimetres.
        print(f"  Storage values fit the [-1, +1] envelope. Two possibilities:")
        print(f"    · MotionBERT/MotionAGFormer convention (camera-space metres,")
        print(f"      root-relative): then \033[1m{convention_mpjpe:.2f} mm\033[0m IS the real MPJPE.")
        print(f"    · Image-pixel normalisation (joint3d_image / res_w · 2):")
        print(f"      then × 1000 is NOT mm; needs camera deprojection.")
        print()
        print(f"  How to disambiguate:")
        print(f"    1. Check the data preprocessing script that built")
        print(f"       {cfg.data_root}{os.sep}{cfg.subset_list[0]}{os.sep}test{os.sep}")
        print(f"       — if it called `joint_cam / 1000.0` or similar before saving,")
        print(f"       you're in metres → (B) is correct mm.")
        print(f"    2. Independently: a *typical* H3.6M MPJPE for a competitive")
        print(f"       lifter with CPN 2D is 38–50 mm. If (B) gives 30–60, the")
        print(f"       convention is consistent. If (B) gives 5 or 500, something")
        print(f"       is off by a factor of 100×.")
    elif unit_class == "millimetres":
        print(f"  Storage is already in mm. The real MPJPE is \033[1m{raw_mpjpe:.2f} mm\033[0m.")
        print(f"  train.py's × 1000 over-scales by 1000× — its 'mm' label is wrong.")
    elif unit_class == "pixels":
        print(f"  Storage is in pixel space. (B) = {convention_mpjpe:.2f} is NOT mm.")
        print(f"  Need camera intrinsics to deproject before reporting MPJPE.")
    else:
        print(f"  Could not auto-classify. Inspect the per-channel stats above and")
        print(f"  cross-reference with the dataset preprocessing pipeline.")
    print()
    print(f"  train.py reported best_mpjpe = {ckpt.get('best_mpjpe', '?')}")
    print(f"  This sweep raw mean         = {raw_mpjpe:.6f}")
    print(f"  Difference (should be ~0)   = "
          f"{abs(float(ckpt.get('best_mpjpe', 0.0)) - raw_mpjpe):.6f}")
    print()


if __name__ == "__main__":
    main()
