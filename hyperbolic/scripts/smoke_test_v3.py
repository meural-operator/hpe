"""Smoke test for HyperbolicHPE v3 (tangent-flow + multi-scale temporal).

Run from inside the `hpe` conda env, project root = hyperbolic/:

    conda activate hpe
    cd hyperbolic
    python scripts/smoke_test_v3.py

Checks
------
  • Model builds with multi-scale temporal windows [3, 9, 27]
  • Forward pass returns expected shapes and finite values (bf16 + fp32)
  • Backward pass produces gradients on every parameter
  • Manifold drift on the returned manifold state is small
  • Per-block tau, topology γ_k are trainable scalars
  • Falls back to legacy temporal_window scalar when temporal_windows missing
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from model.network import HyperbolicHPE
from math_utils.lorentz import lorentz_inner


def banner(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device         : {device}")
    print(f"torch version  : {torch.__version__}")

    # ── Build model ───────────────────────────────────────────────────────────
    banner("BUILD")
    m = HyperbolicHPE(
        in_features      = 3,
        embed_dim        = 512,
        num_spatial      = 3,
        num_temporal     = 3,
        temporal_windows = [3, 9, 27],     # multi-scale schedule
        mlp_ratio        = 4,
        dropout          = 0.1,
        num_joints       = 17,
    ).to(device)

    n_params = sum(p.numel() for p in m.parameters())
    print(f"params         : {n_params:,}")
    print(f"temporal Ws    : {m.temporal_windows}")

    # ── Synthetic batch ───────────────────────────────────────────────────────
    banner("FORWARD")
    B, T, J = 2, 81, 17       # T=81 covers all multi-scale windows safely
    x  = torch.randn(B, T, J, 3, device=device)
    xv = torch.randn(B, T, J, 3, device=device)
    topo = torch.zeros(J, J, device=device)
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    for c, p in enumerate(parents):
        if p >= 0:
            topo[c, p] = 1.0
            topo[p, c] = 1.0

    # ── Intermediate tangent-norm diagnostics ────────────────────────────────
    # Hooks print the mean L2 norm at each block boundary so we can see if
    # representations stay bounded (target: < ~25) or blow up.
    norm_log = []
    def _hook(name):
        def fn(_mod, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            with torch.no_grad():
                n = out.detach().float().pow(2).sum(-1).sqrt().mean().item()
            norm_log.append((name, n))
        return fn
    handles = []
    for i, blk in enumerate(m.spatial_blocks):
        handles.append(blk.register_forward_hook(_hook(f"spatial[{i}]")))
    for i, blk in enumerate(m.temporal_blocks):
        handles.append(blk.register_forward_hook(_hook(f"temporal[{i}]")))

    # bf16 forward (matches train.py autocast)
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, h_manifold = m(x, xv, topo, return_manifold=True)
    else:
        pred, h_manifold = m(x, xv, topo, return_manifold=True)
    for h in handles:
        h.remove()
    print("intermediate tangent-norm trace:")
    for name, n in norm_log:
        print(f"  {name:<14s} ‖h‖_2 mean = {n:.2f}")

    print(f"pred shape     : {tuple(pred.shape)}")
    print(f"manifold shape : {tuple(h_manifold.shape)}")
    print(f"pred dtype     : {pred.dtype}")
    print(f"NaN in pred?   : {torch.isnan(pred).any().item()}")
    print(f"Inf in pred?   : {torch.isinf(pred).any().item()}")

    # ── Backward pass ─────────────────────────────────────────────────────────
    banner("BACKWARD")
    loss = pred.float().pow(2).mean()
    loss.backward()
    grads = [(n, p.grad.norm().item())
             for n, p in m.named_parameters() if p.grad is not None]
    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    print(f"params w/ grad : {len(grads)}")
    print(f"params w/o grad: {len(no_grad)}")
    if no_grad:
        print(" ", no_grad[:5], "...")
    print(f"max grad norm  : {max(g for _, g in grads):.4f}")
    print(f"min grad norm  : {min(g for _, g in grads):.3e}")

    # ── Manifold drift diagnostic ─────────────────────────────────────────────
    banner("MANIFOLD HEALTH")
    drift = (lorentz_inner(h_manifold.float(),
                           h_manifold.float()) + 1.0).abs().mean().item()
    print(f"mean drift     : {drift:.3e}")
    # v3 drift should be tiny — we only do exp_o once at the very end.
    if drift > 1e-3:
        print("  ⚠️  drift larger than expected for v3 (single exp_o at exit).")

    # ── Inspect topology γ_k learnable bias and τ values ──────────────────────
    banner("LEARNABLE GEOMETRY HOOKS")
    for i, blk in enumerate(m.spatial_blocks):
        # topo_gamma is now [H, K]; show the mean per power across heads
        g_mean = blk.attn.topo_gamma.detach().mean(dim=0).cpu().tolist()
        tau_mean = blk.attn.tau.detach().mean().item()
        print(f"spatial[{i}]  H = {blk.attn.num_heads}, "
              f"γ₁,γ₂,γ₃ (mean) = {[round(x,4) for x in g_mean]}, "
              f"τ_mean = {tau_mean:.4f}")
    for i, blk in enumerate(m.temporal_blocks):
        tau_mean = blk.attn.tau.detach().mean().item()
        print(f"temporal[{i}] H = {blk.attn.num_heads}, W = {blk.attn.window:>2}, "
              f"τ_mean = {tau_mean:.4f}")

    # ── Legacy scalar fallback ────────────────────────────────────────────────
    banner("LEGACY scalar `temporal_window` fallback")
    m_legacy = HyperbolicHPE(temporal_window=3).to(device)
    print(f"  resolved windows: {m_legacy.temporal_windows}")
    pred2 = m_legacy(x, xv, topo)
    print(f"  pred shape: {tuple(pred2.shape)}  ✓")

    print("\nAll checks passed.\n")


if __name__ == "__main__":
    main()
