"""
eval_h36m.py — Proper H3.6M evaluation following MotionAGFormer protocol.

This script uses DataReaderH36M to:
  1. Denormalize predictions back to 2.5D image coordinates.
  2. Apply per-clip 2.5d_factor to convert to true millimeter scale.
  3. Compute Protocol #1 (MPJPE) and Protocol #2 (P-MPJPE) per action.

This matches the exact evaluation used in MotionAGFormer/MotionBERT papers.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

try:
    from easydict import EasyDict as edict
except ImportError:
    class edict(dict):
        def __getattr__(self, k): return self[k]
        def __setattr__(self, k, v): self[k] = v

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.network import HyperbolicHPE
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from utils.data import flip_data


def generate_topology_matrix(num_joints, device):
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    topo = torch.zeros((num_joints, num_joints), device=device)
    for child, parent in enumerate(parents):
        if parent >= 0: topo[child, parent] = topo[parent, child] = 1.0
    return topo


def compute_kinematics(x):
    v = torch.zeros_like(x)
    T = x.shape[1]
    if T > 2:
        v[:, 1:-1] = (x[:, 2:] - x[:, :-2]) * 0.5
        v[:, 0] = x[:, 1] - x[:, 0]
        v[:, -1] = x[:, -1] - x[:, -2]
    elif T > 1:
        v[:, 1:] = x[:, 1:] - x[:, :-1]
        v[:, 0] = v[:, 1]
    return v


def run_evaluation(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {opts.checkpoint}")
    ckpt = torch.load(opts.checkpoint, map_location="cpu", weights_only=False)
    
    with open(opts.config) as f:
        args = edict(yaml.safe_load(f))
        
    # Standard evaluation settings
    args.flip = False
    args.use_proj_as_2d = False
    args.add_velocity = False

    model = HyperbolicHPE(
        in_features=3, embed_dim=args.embed_dim,
        num_spatial=args.num_spatial, num_temporal=args.num_temporal,
        num_heads=getattr(args, "num_heads", 8),
        temporal_window=args.temporal_window,
        temporal_windows=args.get("temporal_windows", None),
        mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        num_joints=args.num_joints,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    topo_bias = generate_topology_matrix(args.num_joints, device)

    # ── DataReaderH36M for denormalization + GT in mm ────────────────────────
    print("Initializing DataReaderH36M...")
    datareader = DataReaderH36M(
        n_frames=args.n_frames, 
        sample_stride=1, 
        data_stride_train=args.n_frames // 3, 
        data_stride_test=args.n_frames, 
        dt_root=opts.data_root,
        dt_file=opts.dt_file,
    )

    # ── Test dataset (same as training pipeline) ─────────────────────────────
    test_dataset = MotionDataset3D(args, args.subset_list, "test", return_stats=False)
    dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

    # ── Collect all predictions ──────────────────────────────────────────────
    tta = getattr(opts, 'tta', False)
    print(f"Running inference on {len(test_dataset)} clips... (TTA={'ON' if tta else 'OFF'})")
    results_all = []
    with torch.no_grad():
        for x, y_norm in tqdm(dataloader, desc="Inference"):
            x = x.to(device)
            x_vel = compute_kinematics(x)
            pred_1 = model(x, x_vel, topo_bias)

            if tta:
                x_flip = flip_data(x)
                x_flip_vel = compute_kinematics(x_flip)
                pred_flip = model(x_flip, x_flip_vel, topo_bias)
                pred_2 = flip_data(pred_flip)  # flip back
                pred = (pred_1 + pred_2) / 2.0
            else:
                pred = pred_1
            
            if args.root_rel:
                pred[:, :, 0, :] = 0  # Zero root joint (same as MotionAGFormer)

            results_all.append(pred.cpu().numpy())

    results_all = np.concatenate(results_all)
    print(f"Total prediction clips: {results_all.shape}")

    # ── Denormalize predictions (reverse the /res_w * 2 normalization) ────────
    results_all = datareader.denormalize(results_all)

    # ── Load GT and per-clip metadata from the raw pickle ────────────────────
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]

    assert len(results_all) == len(action_clips), \
        f"Mismatch: {len(results_all)} predictions vs {len(action_clips)} GT clips"

    # ── Per-action evaluation (exact MotionAGFormer protocol) ────────────────
    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    
    results = {}
    results_procrustes = {}
    results_acceleration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_acceleration[action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']

    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]   # (T, 1, 1)
        gt = gt_clips[idx]
        pred = results_all[idx]
        
        # Apply 2.5d factor to convert to true millimeters
        pred *= factor

        # Root-relative alignment
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]

        err1 = calculate_mpjpe(pred, gt)
        e1_all[frame_list] += err1
        
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        
        oc[frame_list] += 1

    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
            if idx < num_test_frames - 2:
                acc_err = acc_err_all[idx] / oc[idx]
                results_acceleration[action].append(acc_err)

    final_result = []
    final_result_procrustes = []
    final_result_acceleration = []
    
    print("\n" + "=" * 65)
    print(f"  {'Action':<20} {'MPJPE (mm)':>12} {'P-MPJPE (mm)':>14}")
    print("-" * 65)
    for action in action_names:
        e1 = np.mean(results[action])
        e2 = np.mean(results_procrustes[action])
        final_result.append(e1)
        final_result_procrustes.append(e2)
        if results_acceleration[action]:
            final_result_acceleration.append(np.mean(results_acceleration[action]))
        print(f"  {action:<20} {e1:>12.1f} {e2:>14.1f}")
    
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    acc_err = np.mean(np.array(final_result_acceleration)) if final_result_acceleration else 0.0

    print("-" * 65)
    print(f"  {'AVERAGE':<20} {e1:>12.1f} {e2:>14.1f}")
    print("=" * 65)
    print(f"\n  Protocol #1 (MPJPE)         : {e1:.1f} mm")
    print(f"  Protocol #2 (P-MPJPE)       : {e2:.1f} mm")
    print(f"  Acceleration Error          : {acc_err:.1f} mm/s²")
    print("=" * 65)

    return e1, e2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H3.6M Evaluation — MotionAGFormer Protocol")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pth checkpoint")
    parser.add_argument("--config", type=str, default="configs/hyperbolic_hpe.yaml", help="Path to config YAML")
    parser.add_argument("--data_root", type=str, default="C:/Users/DIAT/ashish/hpe/steins_shrinkage/MotionAGFormer/data/motion3d",
                        help="Root directory containing the H36M pickle")
    parser.add_argument("--dt_file", type=str, default="h36m_sh_conf_cam_source_final.pkl",
                        help="H36M pickle filename (must contain 2.5d_factor and joints_2.5d_image)")
    parser.add_argument("--tta", action="store_true", help="Enable test-time flip augmentation")
    parser.add_argument("--batch-size", type=int, default=32)
    opts = parser.parse_args()
    
    run_evaluation(opts)