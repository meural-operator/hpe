import torch
import pytest
from model.attention import HyperbolicKinematicAttention
from math_utils.lorentz import project, lorentz_inner

def test_hyperbolic_attention():
    embed_dim = 16
    B, N = 2, 5
    attn = HyperbolicKinematicAttention(embed_dim)
    
    # create synthetic manifold points
    x_euclid = torch.randn(B, N, embed_dim + 1)
    x = project(x_euclid)
    
    # create synthetic tangent velocities (orthogonal to x)
    v_raw = torch.randn(B, N, embed_dim + 1)
    inner = lorentz_inner(v_raw, x, keepdim=True)
    x_vel = v_raw + inner * x
    
    # ensure it's tangent
    assert torch.allclose(lorentz_inner(x_vel, x), torch.tensor(0.0), atol=1e-4)
    
    # adjacency topo
    topo = torch.randint(0, 2, (N, N)).float()
    
    z, weights = attn(x, x_vel, topo)
    
    assert z.shape == (B, N, embed_dim + 1)
    assert weights.shape == (B, N, N)
    
    # output should be on manifold
    assert torch.allclose(lorentz_inner(z, z), torch.tensor(-1.0).expand(B, N), atol=1e-3)
    # weights sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.tensor(1.0).expand(B, N), atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
