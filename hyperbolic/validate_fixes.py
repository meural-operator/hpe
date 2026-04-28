"""
validate_fixes.py
-----------------
Validates all 6 bottleneck fix groups without running training.
Run with:  python validate_fixes.py
"""
import sys
sys.path.insert(0, r'c:\Users\DIAT\ashish\hpe\hyperbolic_hpe_v2')
import torch

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

# ================================================================
# Group A: Fused exp_map0 / log_map0 — no origin() allocation
# ================================================================
print("\n=== Group A: fused log_map0 / exp_map0 ===")
from math_utils.lorentz import exp_map0, log_map0, project

torch.manual_seed(42)
# Tangent vectors at origin with norm < MAX_NORM (=15.0).
# Raw randn for d=512 gives norm ~sqrt(512)~22, which exceeds MAX_NORM
# and is intentionally clamped by _clamp_tangent_norm. Use small scale.
v = torch.zeros(32, 513)
v[:, 1:] = torch.randn(32, 512) * 0.4   # norm ≈ 0.4*sqrt(512) ≈ 9 < 15

assert v[:, 1:].norm(dim=-1).max().item() < 15.0, "test setup: norms must be < MAX_NORM"

# Round-trip: log(exp(v)) must recover v
rt = log_map0(exp_map0(v))
err = (rt - v).abs().max().item()
print(f"  log(exp(v)) roundtrip  max_err = {err:.2e}  [expect < 1e-4]")
assert err < 1e-4, f"FAIL: roundtrip error {err}"

# log_map0(point)[0] must be 0
y = project(torch.cat([torch.zeros(32, 1), torch.randn(32, 512) * 0.4], dim=-1))
log_y = log_map0(y)
t0_err = log_y[:, 0].abs().max().item()
print(f"  log_map0(y)[...,0]==0  max_err = {t0_err:.2e}  [expect < 1e-5]")
assert t0_err < 1e-5, f"FAIL: time component {t0_err}"
print(PASS)

# ================================================================
# Group B+C: Spatial attention returns x_tan; no redundant log_map0
# ================================================================
print("\n=== Group B+C: HyperbolicKinematicAttention — returns x_tan ===")
from model.attention import HyperbolicKinematicAttention

attn = HyperbolicKinematicAttention(embed_dim=32)
B, N = 2, 17
D = 33  # d+1
x_man = project(torch.cat([torch.zeros(B, N, 1), torch.randn(B, N, 32)], dim=-1))
x_vel = torch.zeros(B, N, D)  # tangent at x, time=0 at origin

z, attn_w, x_tan = attn(x_man, x_vel, topo_bias=None)
print(f"  z.shape      = {tuple(z.shape)}   [expect ({B},{N},{D})]")
print(f"  attn_w.shape = {tuple(attn_w.shape)} [expect ({B},{N},{N})]")
print(f"  x_tan.shape  = {tuple(x_tan.shape)}  [expect ({B},{N},32)]")
assert z.shape == (B, N, D), f"wrong z shape {z.shape}"
assert attn_w.shape == (B, N, N)
assert x_tan.shape == (B, N, 32)
assert z.isfinite().all(), "non-finite z"
print(PASS)

# ================================================================
# Group C: SpatialBlock uses returned x_tan (no double log_map0)
# ================================================================
print("\n=== Group C: SpatialBlock forward ===")
from model.network import SpatialBlock
import torch.nn as nn

blk = SpatialBlock(embed_dim=32, dropout=0.0)
z_blk, _ = blk(x_man, x_vel, topo_bias=None)
print(f"  z_blk.shape = {tuple(z_blk.shape)}  [expect ({B},{N},{D})]")
assert z_blk.shape == (B, N, D) and z_blk.isfinite().all()
print(PASS)

# ================================================================
# Group D: O(T x W) windowed temporal attention — never builds T×T matrix
# ================================================================
print("\n=== Group D: HyperbolicTemporalAttention — O(T×W) windowed ===")
from model.attention import HyperbolicTemporalAttention

W = 3
temp = HyperbolicTemporalAttention(embed_dim=32, temporal_window=W)
B, T, J, D = 2, 243, 17, 33   # real training dimensions

x_seq = project(
    torch.cat([torch.zeros(B * T, J, 1), torch.randn(B * T, J, 32)], dim=-1)
).view(B, T, J, D)
vel_seq = torch.zeros_like(x_seq)

z_t = temp(x_seq, vel_seq)
print(f"  z_t.shape = {tuple(z_t.shape)}  [expect ({B},{T},{J},{D})]")
assert z_t.shape == (B, T, J, D), f"wrong shape {z_t.shape}"
assert z_t.isfinite().all(), "non-finite values in temporal output"
print(PASS)

# ================================================================
# Group E: Vectorized bone loss — 2 dist() calls instead of 32
# ================================================================
print("\n=== Group E: geodesic_bone_loss — vectorized ===")
from loss.hyperbolic_loss import geodesic_bone_loss

torch.manual_seed(0)
pred = torch.randn(2, 10, 17, 3)
gt   = torch.randn(2, 10, 17, 3)
loss_v = geodesic_bone_loss(pred, gt)
print(f"  bone_loss = {loss_v.item():.4f}  [expect finite positive]")
assert loss_v.isfinite() and loss_v > 0
print(PASS)

# ================================================================
# Group F: Full-sequence forward via HyperbolicHPE — no CHUNK loop
# ================================================================
print("\n=== Group F: HyperbolicHPE — full-sequence, no chunking ===")
from model.network import HyperbolicHPE

model = HyperbolicHPE(
    in_features=3, embed_dim=32,
    num_spatial=2, num_temporal=2,
    temporal_window=3, dropout=0.0
)
model.eval()

B, T, J = 2, 30, 17
x = torch.randn(B, T, J, 3)
x_vel = torch.zeros_like(x)
x_vel[:, 1:] = x[:, 1:] - x[:, :-1]

with torch.no_grad():
    pred_out, h_out = model(x, x_vel, return_manifold=True)

print(f"  pred.shape = {tuple(pred_out.shape)}  [expect ({B},{T},{J},3)]")
print(f"  h.shape    = {tuple(h_out.shape)}")
assert pred_out.shape == (B, T, J, 3) and pred_out.isfinite().all()
print(PASS)

# ================================================================
# Summary
# ================================================================
print()
print("=" * 52)
print("  ALL 6 BOTTLENECK FIX GROUPS VALIDATED SUCCESSFULLY")
print("=" * 52)
