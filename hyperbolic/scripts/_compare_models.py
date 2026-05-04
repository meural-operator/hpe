"""Compare parameter counts: HyperbolicHPE vs MotionAGFormer"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from model.network import HyperbolicHPE

# --- HyperbolicHPE ---
with open("configs/hyperbolic_hpe.yaml") as f:
    args = yaml.safe_load(f)

model_h = HyperbolicHPE(
    in_features=3, embed_dim=args['embed_dim'],
    num_spatial=args['num_spatial'], num_temporal=args['num_temporal'],
    num_heads=args.get('num_heads', 8),
    temporal_window=args['temporal_window'],
    temporal_windows=args.get('temporal_windows', None),
    mlp_ratio=args['mlp_ratio'], dropout=args['dropout'],
    num_joints=args['num_joints'],
)
n_hyp = sum(p.numel() for p in model_h.parameters())
n_hyp_train = sum(p.numel() for p in model_h.parameters() if p.requires_grad)

print("=" * 60)
print("HyperbolicHPE")
print(f"  Total params:     {n_hyp:>12,}")
print(f"  Trainable params: {n_hyp_train:>12,}")
print(f"  embed_dim={args['embed_dim']}, spatial={args['num_spatial']}, temporal={args['num_temporal']}")
print(f"  heads={args.get('num_heads',8)}, mlp_ratio={args['mlp_ratio']}")

# --- MotionAGFormer ---
sys.path.insert(0, r"C:\Users\DIAT\ashish\hpe\steins_shrinkage\MotionAGFormer")
try:
    from utils.learning import load_model
    from utils.tools import get_config
    
    for variant in ['MotionAGFormer-base', 'MotionAGFormer-small', 'MotionAGFormer-xsmall']:
        try:
            cfg = get_config(f"C:/Users/DIAT/ashish/hpe/steins_shrinkage/MotionAGFormer/configs/h36m/{variant}.yaml")
            model_m = load_model(cfg)
            n_mag = sum(p.numel() for p in model_m.parameters())
            print(f"\n{variant}")
            print(f"  Total params:     {n_mag:>12,}")
            print(f"  embed_dim={getattr(cfg, 'dim_feat', '?')}, depth={getattr(cfg, 'num_block', '?')}")
        except Exception as e:
            print(f"\n{variant}: {e}")
except Exception as e:
    print(f"\nMotionAGFormer import failed: {e}")

print("=" * 60)
