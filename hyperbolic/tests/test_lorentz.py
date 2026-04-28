import torch
import pytest
from math_utils.lorentz import (
    lorentz_inner, dist, origin, project, exp_map, log_map, 
    parallel_transport, einstein_midpoint
)

def test_inner_product():
    x = torch.tensor([2.0, 1.0, 1.0, 1.0])
    y = torch.tensor([3.0, 2.0, 1.0, 2.0])
    # <x, y>_L = -6 + 2 + 1 + 2 = -1
    assert torch.allclose(lorentz_inner(x, y), torch.tensor(-1.0))

def test_project():
    x = torch.tensor([0.0, 0.5, 0.5])
    px = project(x)
    assert px[0] > 0
    # check <px, px> = -1
    assert torch.allclose(lorentz_inner(px, px), torch.tensor(-1.0))

def test_exp_log_map():
    o = origin((3,))
    # tangent vector at o must have v_0 = 0
    v = torch.tensor([0.0, 0.5, -0.5])
    
    # exp
    p = exp_map(o, v)
    assert torch.allclose(lorentz_inner(p, p), torch.tensor(-1.0))
    
    # log
    v_rec = log_map(o, p)
    assert torch.allclose(v, v_rec, atol=1e-5)

def test_parallel_transport():
    x = project(torch.tensor([0.0, 0.1, 0.2]))
    y = project(torch.tensor([0.0, -0.1, 0.5]))
    
    # v in T_x: <v, x> = 0.
    v = torch.tensor([0.5, 1.0, 0.0])
    v = v + lorentz_inner(v, x) * x 
    assert torch.allclose(lorentz_inner(v, x), torch.tensor(0.0), atol=1e-5)
    
    # transport
    v_trans = parallel_transport(x, y, v)
    
    # v_trans should be in T_y
    assert torch.allclose(lorentz_inner(v_trans, y), torch.tensor(0.0), atol=1e-5)
    
    # parallel transport preserves inner product (norm)
    assert torch.allclose(lorentz_inner(v, v), lorentz_inner(v_trans, v_trans), atol=1e-5)

def test_einstein_midpoint():
    x = project(torch.rand((2, 5, 4)))
    w = torch.softmax(torch.rand((2, 5)), dim=-1)
    
    agg = einstein_midpoint(w, x)
    assert agg.shape == (2, 4)
    assert torch.allclose(lorentz_inner(agg, agg), torch.tensor(-1.0).expand(2), atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
