"""
LorentzHPE — True Hyperbolic Human Pose Estimation Network (v4).

Architectural changes from v3:
  1. All layers operate directly on the Lorentz manifold H^d (no tangent-space
     linear layers). Uses LorentzLinear which implements the full Lorentz
     transformation (rotation + boost) per Chen et al. 2022.
  2. embed_dim reduced from 512 → 128 (hyperbolic geometry provides
     exponential capacity per dimension, so fewer dims needed).
  3. 12 interleaved Spatial+Temporal blocks (vs 3+3 in v3).
  4. O(T) linear temporal attention with global receptive field.
  5. Spatial attention retains kinematic velocity + topology bias (our
     unique contribution).

Pipeline:
    2D keypoints → LorentzEmbedding → H^d
                 → 12 × (LorentzSpatialBlock → LorentzTemporalBlock)
                 → log_map0 → PerJointHead → 3D coordinates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lorentz_layers import (
    LorentzLinear, LorentzFFN, LorentzLayerNorm, LorentzActivation,
    lorentz_residual,
)
from model.lorentz_attention import (
    LorentzKinematicAttention,
    LorentzLinearTemporalAttention,
)
from math_utils.lorentz import project, log_map0, EPS


# ─────────────────────────────────────────────────────────────────────────────
# Lorentz Embedding — maps 2D keypoints to H^d
# ─────────────────────────────────────────────────────────────────────────────
class LorentzEmbedding(nn.Module):
    """Maps 2D keypoints (x, y, conf) to points on H^d.

    Position: Linear(2→d) → confidence gate → project to H^d.
    Velocity: Linear(2→d) → stays in tangent space (Euclidean).
    """

    def __init__(self, in_features=3, embed_dim=128):
        super().__init__()
        assert in_features == 3, f"Expected (x, y, conf), got in_features={in_features}"
        self.embed_dim = embed_dim
        self.embed_pos = nn.Linear(2, embed_dim)
        self.embed_vel = nn.Linear(2, embed_dim)
        self.conf_gain = nn.Parameter(torch.tensor(1.0))
        self.conf_bias = nn.Parameter(torch.tensor(0.0))

    def position(self, x):
        """x: [..., 3] → [..., d+1] on H^d."""
        xy, conf = x[..., :2], x[..., 2:3]
        v = self.embed_pos(xy)
        gate = 1.0 + torch.tanh(self.conf_gain * conf + self.conf_bias)
        v = v * gate  # [..., d] spatial components
        # Project to hyperboloid: t = sqrt(1 + ||v||²)
        t = torch.sqrt(1.0 + (v * v).sum(-1, keepdim=True).clamp(min=EPS))
        return torch.cat([t, v], dim=-1)  # [..., d+1]

    def velocity(self, x_vel):
        """x_vel: [..., 3] → [..., d] tangent vector (Euclidean)."""
        return self.embed_vel(x_vel[..., :2])

    def forward(self, x):
        return self.position(x)


# ─────────────────────────────────────────────────────────────────────────────
# Per-joint output head (same as v3, operates on tangent/Euclidean vectors)
# ─────────────────────────────────────────────────────────────────────────────
class PerJointHead(nn.Module):
    """Decodes each joint independently from its d-dim tangent representation.

    Weights shape: [J, d, 3], biases [J, 3].
    """

    def __init__(self, embed_dim, num_joints, hidden_ratio=1):
        super().__init__()
        self.J = num_joints
        self.norm = nn.LayerNorm(embed_dim)
        hidden = embed_dim * hidden_ratio
        self.fc1_w = nn.Parameter(torch.empty(num_joints, embed_dim, hidden))
        self.fc1_b = nn.Parameter(torch.zeros(num_joints, hidden))
        self.fc2_w = nn.Parameter(torch.empty(num_joints, hidden, 3))
        self.fc2_b = nn.Parameter(torch.zeros(num_joints, 3))
        self.act = nn.GELU()
        nn.init.kaiming_uniform_(self.fc1_w, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.fc2_w, a=5 ** 0.5)

    def forward(self, x):
        """x: [B, T, J, d] → [B, T, J, 3]."""
        x = self.norm(x)
        h = torch.einsum('btjd,jdh->btjh', x, self.fc1_w) + self.fc1_b
        h = self.act(h)
        y = torch.einsum('btjh,jho->btjo', h, self.fc2_w) + self.fc2_b
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Spatial Block (on-manifold)
# ─────────────────────────────────────────────────────────────────────────────
class LorentzSpatialBlock(nn.Module):
    """Spatial reasoning block: joints attend to each other within a frame.

    Pre-norm on manifold → LorentzKinematicAttention → residual → FFN.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = LorentzLayerNorm(embed_dim)
        self.attn = LorentzKinematicAttention(embed_dim, num_heads=num_heads,
                                              dropout=dropout)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm2 = LorentzLayerNorm(embed_dim)
        self.ffn = LorentzFFN(embed_dim, mlp_ratio, dropout)

    def forward(self, x, v_tan, topo_bias):
        """
        x:      [B, J, d+1]  on H^d
        v_tan:  [B, J, d]    velocity (tangent/Euclidean)
        topo:   [J, J]       adjacency
        Returns: [B, J, d+1] on H^d
        """
        # Attention with pre-norm
        z, _ = self.attn(self.norm1(x), self.norm_v(v_tan), topo_bias)
        h = lorentz_residual(x, z)

        # FFN with pre-norm (FFN has built-in residual via FHNN style)
        h = self.ffn(self.norm2(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Block (on-manifold, O(T) linear attention)
# ─────────────────────────────────────────────────────────────────────────────
class LorentzTemporalBlock(nn.Module):
    """Temporal reasoning block: same joint attends across all frames.

    Uses O(T) linear attention for global temporal receptive field.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = LorentzLayerNorm(embed_dim)
        self.attn = LorentzLinearTemporalAttention(embed_dim, num_heads=num_heads,
                                                    dropout=dropout)
        self.norm2 = LorentzLayerNorm(embed_dim)
        self.ffn = LorentzFFN(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """x: [B, T, J, d+1] on H^d → [B, T, J, d+1] on H^d."""
        z = self.attn(self.norm1(x))
        h = lorentz_residual(x, z)
        h = self.ffn(self.norm2(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Main Network: LorentzHPE
# ─────────────────────────────────────────────────────────────────────────────
class LorentzHPE(nn.Module):
    """True Hyperbolic HPE — v4.

    All hidden representations live on the Lorentz hyperboloid H^d.
    Output is decoded via log_map0 → PerJointHead → R³.
    """

    def __init__(
        self,
        in_features=3,
        embed_dim=128,
        n_layers=12,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        num_joints=17,
        # Legacy compat args (ignored in v4)
        temporal_window=None,
        temporal_windows=None,
        num_spatial=None,
        num_temporal=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.n_layers = n_layers

        # Embedding: 2D keypoints → H^d
        self.embed = LorentzEmbedding(in_features, embed_dim)

        # Joint identity embedding (added to spatial components on manifold)
        self.joint_embed = nn.Parameter(torch.zeros(num_joints, embed_dim))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        # Interleaved Spatial + Temporal blocks
        self.spatial_blocks = nn.ModuleList([
            LorentzSpatialBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.temporal_blocks = nn.ModuleList([
            LorentzTemporalBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Decoder: log_map0 → PerJointHead
        self.head = PerJointHead(embed_dim, num_joints)

    def forward(self, x, x_vel, topo_bias=None, return_manifold=False):
        """
        x       : [B, T, J, 3]     2D keypoints (x, y, conf)
        x_vel   : [B, T, J, 3]     finite-difference velocities
        topo_bias : [J, J]          kinematic adjacency
        Returns : [B, T, J, 3]     predicted 3D coordinates
        """
        B, T, J, _ = x.shape
        d = self.embed_dim

        # ── Embed to manifold ────────────────────────────────────────────────
        h = self.embed.position(x)        # [B, T, J, d+1] on H^d
        v = self.embed.velocity(x_vel)    # [B, T, J, d] tangent (Euclidean)

        # Add joint identity to spatial components, re-project
        h_spatial = h[..., 1:] + self.joint_embed.view(1, 1, J, d)
        h_time = torch.sqrt(
            1.0 + (h_spatial * h_spatial).sum(-1, keepdim=True).clamp(min=EPS)
        )
        h = torch.cat([h_time, h_spatial], dim=-1)  # [B, T, J, d+1]

        # ── Interleaved Spatial → Temporal blocks ────────────────────────────
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: per-frame joint attention (flatten T into batch)
            h_flat = h.reshape(B * T, J, d + 1)
            v_flat = v.reshape(B * T, J, d)
            h_flat = s_block(h_flat, v_flat, topo_bias)
            h = h_flat.reshape(B, T, J, d + 1)

            # Temporal: per-joint cross-frame attention (O(T) linear)
            h = t_block(h)

        # ── Decode: H^d → R³ ────────────────────────────────────────────────
        # Map from manifold to tangent space at origin, take spatial components
        h_tangent = log_map0(h)          # [B, T, J, d+1]
        h_spatial = h_tangent[..., 1:]   # [B, T, J, d]

        pred = self.head(h_spatial)      # [B, T, J, 3]

        if return_manifold:
            return pred, h
        return pred
