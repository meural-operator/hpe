"""Gradient flow test + training step test for LorentzHPE v4."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.lorentz_network import LorentzHPE
from loss.pose3d import loss_mpjpe
from train import generate_topology_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LorentzHPE(
    in_features=3, embed_dim=128, n_layers=12,
    num_heads=8, mlp_ratio=4, dropout=0.1, num_joints=17,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
topo = generate_topology_matrix(17, device)

B, T, J = 2, 27, 17
x = torch.randn(B, T, J, 3, device=device)
x_vel = torch.randn(B, T, J, 3, device=device)
y = torch.randn(B, T, J, 3, device=device)

# Forward
pred = model(x, x_vel, topo)
loss = loss_mpjpe(pred, y)
print(f"Loss: {loss.item():.6f}")

# Backward
loss.backward()

# Check gradients
nan_count = 0
zero_count = 0
total_count = 0
for name, p in model.named_parameters():
    total_count += 1
    if p.grad is None:
        zero_count += 1
    elif torch.isnan(p.grad).any():
        nan_count += 1
        print(f"  NaN grad: {name}")
    elif p.grad.abs().max() == 0:
        zero_count += 1

print(f"\nGradient check: {total_count} params, {nan_count} NaN, {zero_count} zero grad")
if nan_count == 0:
    print("✓ No NaN gradients!")
else:
    print("✗ Found NaN gradients!")

# Optimizer step
optimizer.step()
optimizer.zero_grad()

# Second forward to check stability
pred2 = model(x, x_vel, topo)
loss2 = loss_mpjpe(pred2, y)
print(f"\nAfter 1 step — Loss: {loss2.item():.6f}")
if not torch.isnan(loss2):
    print("✓ Training step stable!")
else:
    print("✗ Training step produced NaN!")
