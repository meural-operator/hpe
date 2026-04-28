"""End-to-end test for Hyperbolic HPE v2: forward, all losses, backward, VRAM."""
import sys
sys.path.insert(0, 'c:/Users/DIAT/ashish/hpe/hyperbolic_hpe_v2')
import torch
from model.network import HyperbolicHPE
from loss.pose3d import loss_mpjpe
from loss.hyperbolic_loss import (
    geodesic_velocity_loss, geodesic_bone_loss,
    manifold_drift_loss, UncertaintyWeightedLoss
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

device = torch.device('cuda')
model = HyperbolicHPE(3, 512, 3, 3, 3, 4, 0.1).to(device)
loss_weighter = UncertaintyWeightedLoss(4).to(device)

print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# Simulate batch=2, chunk=16 frames, 17 joints
B, T, J, C = 2, 16, 17, 3
x = torch.randn(B, T, J, C, device=device)
v = torch.randn(B, T, J, C, device=device) * 0.1
y = torch.randn(B, T, J, C, device=device)
topo = torch.zeros(J, J, device=device)

# Forward with manifold states
pred, h_manifold = model(x, v, topo, return_manifold=True)

# Compute all losses
l_mpjpe = loss_mpjpe(pred, y)
l_vel = geodesic_velocity_loss(pred, y)
l_bone = geodesic_bone_loss(pred, y)
l_drift = manifold_drift_loss(h_manifold.view(-1, h_manifold.shape[-1]))

# Dynamic weighting
loss_total, weights = loss_weighter(l_mpjpe, l_vel, l_bone, l_drift)

# Backward
loss_total.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Pred shape: {pred.shape}")
print(f"H_manifold shape: {h_manifold.shape}")
print(f"Loss total: {loss_total.item():.4f}")
print(f"  MPJPE: {l_mpjpe.item():.4f} (w={weights[0]:.4f})")
print(f"  Vel:   {l_vel.item():.4f} (w={weights[1]:.4f})")
print(f"  Bone:  {l_bone.item():.4f} (w={weights[2]:.4f})")
print(f"  Drift: {l_drift.item():.6f} (w={weights[3]:.4f})")
print(f"NaN: {torch.isnan(loss_total).item()}")
print(f"Peak VRAM: {mem:.2f} GB")
print("V2 END-TO-END OK")
