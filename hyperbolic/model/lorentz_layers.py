"""
Fully Hyperbolic Layers — v4.

Core building blocks that operate DIRECTLY on the Lorentz manifold H^d,
implementing the full Lorentz transformation (rotation + boost).

Based on:
  - Chen et al. 2022 "Fully Hyperbolic Neural Networks" (ACL)
  - Hypformer (Yang et al. KDD 2024) for LayerNorm and activation

Key difference from v3 tangent-space layers:
  v3: exp_map0(nn.Linear(log_map0(x)))  →  rotation only, no boost
  v4: LorentzLinear(x)                  →  full Lorentz transformation

The LorentzLinear applies a standard linear map, then re-projects onto the
hyperboloid by computing the time coordinate via a learned sigmoid-scaled
boost and rescaling spatial coordinates to satisfy ⟨x,x⟩_L = -1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from math_utils.lorentz import exp_map0, log_map0, project, EPS


# ─────────────────────────────────────────────────────────────────────────────
# LorentzLinear — Full Lorentz Transformation (Chen et al. 2022)
# ─────────────────────────────────────────────────────────────────────────────
class LorentzLinear(nn.Module):
    """Fully hyperbolic linear layer operating on Lorentz vectors.

    For input x ∈ H^{d_in} (shape [..., d_in+1]) on the hyperboloid,
    produces output y ∈ H^{d_out} (shape [..., d_out+1]).

    Steps:
        1. z = W @ x + b                    (standard linear, d_in+1 → d_out+1)
        2. t = sigmoid(z[...,0]) * s + 1+ε  (learned boost → time coord > 1)
        3. spatial = z[...,1:] * √((t²-1) / ‖z[...,1:]‖²)  (rescale to H^d)
        4. output = [t, spatial]             (guaranteed on hyperboloid)

    The sigmoid-scaled time coordinate is the "boost" that tangent-space
    linear layers lack — it controls how deep in the hyperbolic tree the
    output point sits.
    """

    def __init__(self, in_features, out_features, bias=True, dropout=0.1):
        """
        Args:
            in_features:  spatial dimension of input  (d_in, NOT d_in+1)
            out_features: spatial dimension of output (d_out, NOT d_out+1)
            bias:         whether to include bias in the linear map
            dropout:      dropout rate applied before the linear map
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Linear operates on full Lorentz vectors: (d_in+1) → (d_out+1)
        self.linear = nn.Linear(in_features + 1, out_features + 1, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # Learnable scale for the sigmoid boost (controls depth range)
        self.scale = nn.Parameter(torch.tensor(2.3))
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights. Zero out the time-coordinate column so that
        at init, the linear doesn't mix time and space unpredictably."""
        stdv = 0.02
        nn.init.uniform_(self.linear.weight, -stdv, stdv)
        # Zero out weights connecting FROM the time coordinate (column 0)
        with torch.no_grad():
            self.linear.weight[:, 0] = 0
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, residual=None):
        """
        Args:
            x:        [..., d_in+1] points on H^{d_in}
            residual: [..., d_out+1] optional residual added before projection
        Returns:
            y:        [..., d_out+1] points on H^{d_out}
        """
        z = self.linear(self.dropout(x))

        # Add residual in ambient space BEFORE Lorentz projection (FHNN style)
        if residual is not None:
            z = z + residual

        # Split time and spatial components
        z_time = z[..., :1]           # [..., 1]
        z_spatial = z[..., 1:]        # [..., d_out]

        # Boost: sigmoid → bounded positive time coordinate > 1
        time = z_time.sigmoid() * self.scale.exp() + 1.0 + EPS  # [..., 1]

        # Rescale spatial to satisfy hyperboloid constraint: -t² + ||s||² = -1
        # → ||s||² = t² - 1
        sq_norm = (z_spatial * z_spatial).sum(dim=-1, keepdim=True).clamp(min=EPS)
        target_sq_norm = (time * time - 1.0).clamp(min=EPS)
        scale_factor = (target_sq_norm / sq_norm).sqrt()
        spatial = z_spatial * scale_factor

        return torch.cat([time, spatial], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# LorentzLayerNorm — Normalization on the hyperboloid
# ─────────────────────────────────────────────────────────────────────────────
class LorentzLayerNorm(nn.Module):
    """Layer normalization on the Lorentz manifold.

    Normalizes the spatial components (index 1:), then recomputes the time
    coordinate to maintain the hyperboloid constraint.

    Following Hypformer's HRC approach:
        1. Normalize spatial: s_norm = LayerNorm(x[..., 1:])
        2. Recompute time: t = sqrt(1 + ||s_norm||²)
        3. Output: [t, s_norm] ∈ H^d
    """

    def __init__(self, d):
        """Args: d = spatial dimension (NOT d+1)."""
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        """x: [..., d+1] on H^d → [..., d+1] on H^d."""
        spatial = x[..., 1:]                            # [..., d]
        spatial_norm = self.norm(spatial)                # [..., d]
        time = torch.sqrt(
            1.0 + (spatial_norm * spatial_norm).sum(dim=-1, keepdim=True).clamp(min=EPS)
        )                                               # [..., 1]
        return torch.cat([time, spatial_norm], dim=-1)   # [..., d+1]


# ─────────────────────────────────────────────────────────────────────────────
# LorentzActivation — On-manifold nonlinearity
# ─────────────────────────────────────────────────────────────────────────────
class LorentzActivation(nn.Module):
    """Apply activation to spatial components, recompute time.

    This preserves the hyperboloid constraint while introducing nonlinearity.
    """

    def __init__(self, act_fn=None):
        super().__init__()
        self.act = act_fn or nn.GELU()

    def forward(self, x):
        """x: [..., d+1] → [..., d+1], both on H^d."""
        spatial = self.act(x[..., 1:])
        time = torch.sqrt(
            1.0 + (spatial * spatial).sum(dim=-1, keepdim=True).clamp(min=EPS)
        )
        return torch.cat([time, spatial], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# LorentzFFN — Fully Hyperbolic Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────
class LorentzFFN(nn.Module):
    """Two-layer FFN operating entirely on the hyperboloid.

    Architecture (FHNN style):
        x → LorentzLinear(d → d*ratio) → LorentzActivation → LorentzLinear(d*ratio → d, residual=x)

    The second LorentzLinear takes the original input as a residual, which is
    added in ambient space before the Lorentz projection — this implements
    the skip connection without leaving the manifold.
    """

    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.fc1 = LorentzLinear(embed_dim, hidden, dropout=dropout)
        self.act = LorentzActivation()
        self.fc2 = LorentzLinear(hidden, embed_dim, dropout=dropout)

    def forward(self, x):
        """x: [..., d+1] on H^d → [..., d+1] on H^d (with skip connection)."""
        h = self.fc1(x)
        h = self.act(h)
        # Residual: add x in ambient space before projection in fc2
        return self.fc2(h, residual=x)


# ─────────────────────────────────────────────────────────────────────────────
# Lorentz Residual — Skip connection on the hyperboloid
# ─────────────────────────────────────────────────────────────────────────────
def lorentz_residual(x, fx):
    """Residual connection on H^d: add in ambient space and re-project.

    This is equivalent to:
        project(x + fx)
    where project() adjusts the time coordinate to satisfy ⟨y,y⟩_L = -1.

    Geometrically, this preserves the spatial (information-carrying) components
    of both x and f(x), and only adjusts the time coordinate.
    """
    combined = x + fx
    return project(combined)
