"""Quick smoke test for LorentzHPE v4 architecture."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.lorentz_network import LorentzHPE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model with v4 config
model = LorentzHPE(
    in_features=3,
    embed_dim=128,
    n_layers=12,
    num_heads=8,
    mlp_ratio=4,
    dropout=0.1,
    num_joints=17,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LorentzHPE v4 | Total params: {n_params:,} | Trainable: {n_train:,}")

# Test forward pass with a small batch
B, T, J = 2, 27, 17  # small batch for speed
x = torch.randn(B, T, J, 3, device=device)
x_vel = torch.randn(B, T, J, 3, device=device)

# Topology bias
from train import generate_topology_matrix
topo = generate_topology_matrix(J, device)

print(f"Input: x={x.shape}, x_vel={x_vel.shape}")

try:
    with torch.no_grad():
        pred = model(x, x_vel, topo)
    print(f"Output: pred={pred.shape}")
    print(f"Output range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print("✓ Forward pass successful!")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

# Test with return_manifold
try:
    with torch.no_grad():
        pred, h_manifold = model(x, x_vel, topo, return_manifold=True)
    print(f"\nManifold output: {h_manifold.shape}")
    # Check hyperboloid constraint: ⟨x,x⟩_L should be ≈ -1
    from math_utils.lorentz import lorentz_inner
    sqnorm = lorentz_inner(h_manifold.reshape(-1, h_manifold.shape[-1]),
                           h_manifold.reshape(-1, h_manifold.shape[-1]))
    drift = (sqnorm + 1.0).abs().mean().item()
    print(f"Manifold drift (should be ~0): {drift:.6f}")
    print("✓ Manifold constraint check passed!")
except Exception as e:
    print(f"✗ Manifold check failed: {e}")
    import traceback
    traceback.print_exc()
