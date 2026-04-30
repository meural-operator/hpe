import functools
import torch
import torch.nn.functional as F

# fp32-safe epsilon. We force every Lorentz primitive to run in fp32 (see
# `_fp32_lorentz` below) so this is honoured even when the surrounding model
# runs under bf16 autocast — bf16 epsilon (~4e-3) would round 1e-7 to zero
# and `clamp(min=1+EPS)` would hit the acosh boundary, blowing up gradients.
EPS = 1e-7

# Maximum norm for exp_map input to prevent cosh/sinh overflow.
# cosh(15) ≈ 1.6e6 → cosh²(15) ≈ 2.7e12: safe in float32 for inner products.
# cosh(50) ≈ 2.6e21 → cosh²(50) ≈ 6.7e42: EXCEEDS float32 precision.
MAX_NORM = 15.0


def _fp32_lorentz(fn):
    """Force a Lorentz primitive to run in fp32 regardless of caller dtype.

    The trig ops here (cosh/sinh/acosh) and the EPS-guarded reciprocals are
    all unsafe in bf16. The matmul-free arithmetic inside is cheap, so the
    upcast cost is negligible vs. the numerical risk of running them in bf16.
    Output is fp32; downstream `nn.Linear` under autocast will recast as needed.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def cast(x):
            if isinstance(x, torch.Tensor) and x.is_floating_point() and x.dtype != torch.float32:
                return x.float()
            return x
        args = tuple(cast(a) for a in args)
        kwargs = {k: cast(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)
    return wrapper


@_fp32_lorentz
def lorentz_inner(u, v, keepdim=False):
    """Lorentzian inner product: <u, v>_L = -u_0*v_0 + sum(u_i*v_i)"""
    uv = u * v
    res = -uv[..., 0] + uv[..., 1:].sum(dim=-1)
    if keepdim:
        res = res.unsqueeze(-1)
    return res


@_fp32_lorentz
def lorentz_sqnorm(u, keepdim=False):
    return lorentz_inner(u, u, keepdim=keepdim)


@_fp32_lorentz
def dist(u, v, keepdim=False):
    """Hyperbolic distance: d_L(u, v) = arccosh(-<u, v>_L)"""
    inner = -lorentz_inner(u, v, keepdim=keepdim)
    # Aggressive clamp: arccosh domain is [1, inf), float32 needs margin
    inner = torch.clamp(inner, min=1.0 + EPS)
    return torch.acosh(inner)


def origin(shape, device=None, dtype=None):
    """Origin on the Lorentz manifold: (1, 0, ..., 0)"""
    o = torch.zeros(shape, device=device, dtype=dtype)
    o[..., 0] = 1.0
    return o


@_fp32_lorentz
def project(x):
    """Projects x ∈ R^{d+1} onto H^d by setting x_0 = sqrt(1 + ||x_{1:}||²)"""
    spatial = x[..., 1:]
    x0 = torch.sqrt(1.0 + torch.sum(spatial ** 2, dim=-1, keepdim=True))
    return torch.cat([x0, spatial], dim=-1)


def _clamp_tangent_norm(v):
    """Clamps the Lorentzian norm of tangent vector to prevent exp_map overflow."""
    sqnorm = lorentz_sqnorm(v, keepdim=True)
    sqnorm = torch.clamp(sqnorm, min=0.0)
    norm = torch.sqrt(sqnorm + EPS)
    
    # If norm > MAX_NORM, scale v down
    scale = torch.clamp(MAX_NORM / (norm + EPS), max=1.0)
    return v * scale, norm * scale


def _spatial_norm(vi):
    """Euclidean norm of spatial (index 1:) components. Returns shape [..., 1]."""
    return torch.sqrt(torch.sum(vi ** 2, dim=-1, keepdim=True).clamp(min=0.0) + EPS)


@_fp32_lorentz
def exp_map(x, v):
    """
    Exponential map: exp_x(v) = cosh(||v||) x + sinh(||v||) v/||v||
    Tangent norm is clamped to prevent cosh/sinh overflow (trust region).
    """
    v_clamped, norm_v = _clamp_tangent_norm(v)
    
    cosh_nv = torch.cosh(norm_v)
    sinh_nv = torch.sinh(norm_v)
    direction = v_clamped / (norm_v + EPS)
    
    res = cosh_nv * x + sinh_nv * direction
    
    # For very small norms, use first-order approximation: exp_x(v) ≈ x + v
    cond = (norm_v > EPS)
    return torch.where(cond, res, x + v_clamped)


@_fp32_lorentz
def exp_map0(v):
    """
    Exponential map from origin — allocation-free fused implementation.

    For tangent vectors at the origin with v[..., 0] == 0, substituting
    o = (1, 0, ..., 0) into exp_map(o, v) gives analytically:
        result[..., 0]  = cosh(||v[1:]||)
        result[..., 1:] = sinh(||v[1:]||) * v[1:] / ||v[1:]||

    Numerically identical to the previous implementation but avoids
    allocating a full origin tensor on every call.
    """
    vi = v[..., 1:]                                    # spatial part [..., d]
    norm = _spatial_norm(vi)                           # [..., 1]
    norm_clamped = torch.clamp(norm, max=MAX_NORM)

    cosh_n = torch.cosh(norm_clamped)                  # [..., 1]
    sinh_n = torch.sinh(norm_clamped)                  # [..., 1]

    x0 = cosh_n                                        # [..., 1]
    xi = (sinh_n / (norm + EPS)) * vi                 # [..., d]
    res = torch.cat([x0, xi], dim=-1)

    # Small-norm fallback: exp_o(v) ≈ o + v = (1, v[1:])
    fallback = torch.cat([torch.ones_like(x0), vi], dim=-1)
    return torch.where(norm > EPS, res, fallback)


@_fp32_lorentz
def log_map(x, y):
    """
    Logarithmic map: log_x(y) = d(x,y) * (y + <x,y>_L x) / ||y + <x,y>_L x||_L
    """
    xy = lorentz_inner(x, y, keepdim=True)
    d = dist(x, y, keepdim=True)
    
    # Numerator: y + <x,y>_L * x (this is the un-normalized tangent direction)
    num = y + xy * x
    sqnorm_num = lorentz_sqnorm(num, keepdim=True)
    sqnorm_num = torch.clamp(sqnorm_num, min=0.0)
    norm_num = torch.sqrt(sqnorm_num + EPS)
    
    # Clamp d to prevent explosion when dividing
    d = torch.clamp(d, max=MAX_NORM)
    
    res = d * (num / (norm_num + EPS))
    cond = (norm_num > EPS)
    return torch.where(cond, res, torch.zeros_like(res))


@_fp32_lorentz
def log_map0(y):
    """
    Logarithmic map to origin — allocation-free fused implementation.

    For any point y = (y0, y[1:]) on H^d, substituting o = (1, 0, ..., 0)
    into log_map(o, y) gives analytically:
        d        = acosh(y0)                    [geodesic distance from origin]
        result[..., 0]  = 0
        result[..., 1:] = d * y[1:] / ||y[1:]||

    Numerically identical to the previous implementation but avoids
    allocating a full origin tensor on every call.
    """
    y0 = y[..., :1]                                    # [..., 1]
    yi = y[..., 1:]                                    # [..., d]

    # Geodesic distance from origin
    d = torch.acosh(torch.clamp(y0, min=1.0 + EPS))   # [..., 1]
    d = torch.clamp(d, max=MAX_NORM)

    norm_yi = _spatial_norm(yi)                        # [..., 1]

    res_0 = torch.zeros_like(y0)                       # [..., 1]
    res_i = (d / (norm_yi + EPS)) * yi                # [..., d]
    res = torch.cat([res_0, res_i], dim=-1)

    # Fallback when y ≈ origin: log_o(o) = 0
    return torch.where(norm_yi > EPS, res, torch.zeros_like(res))


@_fp32_lorentz
def parallel_transport(x, y, v):
    """
    Parallel transport of v ∈ T_x H^d to T_y H^d:
    P_{x→y}(v) = v + <y, v>_L / (1 - <x, y>_L) * (x + y)
    """
    xy = lorentz_inner(x, y, keepdim=True)
    yv = lorentz_inner(y, v, keepdim=True)
    
    # Since <x,y>_L <= -1, we have 1 - <x,y>_L >= 2, but float errors can violate this
    denom = torch.clamp(1.0 - xy, min=EPS)
    
    return v + (yv / denom) * (x + y)


@_fp32_lorentz
def einstein_midpoint(weights, x):
    """
    Einstein Midpoint aggregation: exp_0(Σ w_j log_0(x_j))
    """
    w = weights.unsqueeze(-1)
    log_x = log_map0(x)
    agg_log = torch.sum(w * log_x, dim=-2)
    return exp_map0(agg_log)
