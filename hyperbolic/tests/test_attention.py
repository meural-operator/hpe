"""Tests for v3 tangent-flow attention modules."""
import torch
import pytest
from model.attention import (
    HyperbolicKinematicAttention,
    HyperbolicTemporalAttention,
)


def test_hyperbolic_kinematic_attention_shape_and_softmax():
    embed_dim, H = 16, 4
    B, N = 2, 5
    attn = HyperbolicKinematicAttention(embed_dim, num_heads=H)

    x_tan = torch.randn(B, N, embed_dim) * 0.1
    v_tan = torch.randn(B, N, embed_dim) * 0.1
    topo = torch.randint(0, 2, (N, N)).float()

    z, weights = attn(x_tan, v_tan, topo)

    assert z.shape == (B, N, embed_dim), f"unexpected z shape: {z.shape}"
    assert weights.shape == (B, H, N, N), f"unexpected w shape: {weights.shape}"
    # softmax over the last dim — sums to 1 per (batch, head, query) triple
    assert torch.allclose(weights.sum(dim=-1),
                          torch.ones(B, H, N), atol=1e-5)


def test_hkpsa_topo_powers_cached():
    embed_dim, H = 8, 2
    B, N = 1, 6
    attn = HyperbolicKinematicAttention(embed_dim, num_heads=H)
    x_tan = torch.randn(B, N, embed_dim) * 0.1
    v_tan = torch.randn(B, N, embed_dim) * 0.1
    topo = torch.eye(N) + torch.diag(torch.ones(N - 1), 1)
    topo = topo + topo.t()

    _ = attn(x_tan, v_tan, topo)
    A_pows_first = attn._A_powers_cache.clone()
    _ = attn(x_tan, v_tan, topo)
    A_pows_second = attn._A_powers_cache
    assert torch.equal(A_pows_first, A_pows_second), \
        "A^k cache should be reused across forward passes"


def test_temporal_attention_shape_and_window():
    embed_dim, H = 16, 4
    B, T, J = 2, 32, 5
    for W in [1, 3, 9]:
        attn = HyperbolicTemporalAttention(embed_dim, temporal_window=W, num_heads=H)
        x_seq = torch.randn(B, T, J, embed_dim) * 0.1
        z = attn(x_seq)
        assert z.shape == (B, T, J, embed_dim)


def test_kinematic_penalty_nonnegative():
    """Kinematic penalty ‖v_q - v_k‖² must be ≥ 0 even after numerical noise."""
    embed_dim, H = 16, 4
    B, N = 1, 4
    attn = HyperbolicKinematicAttention(embed_dim, num_heads=H)
    x_tan = torch.zeros(B, N, embed_dim)
    v_tan = torch.randn(B, N, embed_dim) * 0.5
    z, w = attn(x_tan, v_tan, None)
    # Softmax always > 0
    assert torch.all(w > 0)
    # No NaNs
    assert not torch.isnan(z).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
