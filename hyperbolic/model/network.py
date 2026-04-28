import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import HyperbolicKinematicAttention, HyperbolicTemporalAttention
from model.embedding import PhaseSpaceEmbedding
from math_utils.lorentz import log_map0, exp_map0, origin, parallel_transport, lorentz_inner


class TangentFFN(nn.Module):
    """Feed-Forward Network in the tangent space at origin."""

    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_spatial):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x_spatial)))))


class SpatialBlock(nn.Module):
    """Spatial attention block: joints attend to each other within a frame."""

    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = HyperbolicKinematicAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = TangentFFN(embed_dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_vel, topo_bias):
        """x: [B, N, d+1] on manifold"""
        # x_tan is log_map0(x)[..., 1:] — returned by attention to avoid recomputation
        z, attn_w, x_tan = self.attn(x, x_vel, topo_bias)
        z_tan = log_map0(z)[..., 1:]
        h = self.norm1(x_tan + self.drop(z_tan))
        h = self.norm2(h + self.ffn(h))
        h_full = F.pad(h, (1, 0), value=0.0)
        return exp_map0(h_full), attn_w


class TemporalBlock(nn.Module):
    """Temporal attention block: same joint attends across frames."""

    def __init__(self, embed_dim, temporal_window=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = HyperbolicTemporalAttention(embed_dim, temporal_window)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = TangentFFN(embed_dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_seq, vel_seq):
        """
        x_seq: [B, T, J, d+1] manifold points
        vel_seq: [B, T, J, d+1] velocities
        """
        B, T, J, D = x_seq.shape
        z_seq = self.attn(x_seq, vel_seq)

        # Residual in tangent space
        x_tan = log_map0(x_seq.reshape(B*T*J, D))[..., 1:].reshape(B, T, J, -1)
        z_tan = log_map0(z_seq.reshape(B*T*J, D))[..., 1:].reshape(B, T, J, -1)
        d = x_tan.shape[-1]

        h = self.norm1(x_tan + self.drop(z_tan))
        h = self.norm2(h + self.ffn(h))

        h_full = F.pad(h, (1, 0), value=0.0)
        out = exp_map0(h_full.reshape(B*T*J, d+1)).reshape(B, T, J, d+1)
        return out


class HyperbolicHPE(nn.Module):
    """Hyperbolic Kinematic Phase-Space Attention Network v2.

    6-layer interleaved spatial/temporal architecture:
      Spatial → Temporal → Spatial → Temporal → Spatial → Temporal

    Returns both predictions AND intermediate manifold states for drift monitoring.
    """

    def __init__(self, in_features=3, embed_dim=512, num_spatial=3, num_temporal=3,
                 temporal_window=3, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.embed = PhaseSpaceEmbedding(in_features, embed_dim)

        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(embed_dim, mlp_ratio, dropout)
            for _ in range(num_spatial)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(embed_dim, temporal_window, mlp_ratio, dropout)
            for _ in range(num_temporal)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 3)
        )

    def forward(self, x, x_vel, topo_bias=None, return_manifold=False):
        """
        x:     [B, T, J, 3] raw poses across time
        x_vel: [B, T, J, 3] velocities
        topo_bias: [J, J] skeleton adjacency

        Returns:
            pred: [B, T, J, 3] predicted poses
            h_manifold: (optional) [B, T, J, d+1] last hidden manifold state for drift monitoring
        """
        B, T, J, C = x.shape

        # Embed all frames into Lorentz manifold
        x_flat = x.reshape(B * T, J, C)
        h_x = self.embed(x_flat)  # [B*T, J, d+1]
        D = h_x.shape[-1]

        # Embed velocities
        o = origin(h_x.shape, device=h_x.device, dtype=h_x.dtype)
        v_euclid = self.embed.fc(x_vel.reshape(B * T, J, C))
        v_tangent0 = F.pad(v_euclid, (1, 0), value=0.0)
        h_v = parallel_transport(o, h_x, v_tangent0)  # [B*T, J, D]

        # Reshape for interleaved processing
        h_x_seq = h_x.reshape(B, T, J, D)
        h_v_seq = h_v.reshape(B, T, J, D)

        # Interleaved Spatial → Temporal blocks
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: per-frame joint attention
            h_x_flat = h_x_seq.reshape(B * T, J, D)
            h_v_flat = h_v_seq.reshape(B * T, J, D)
            h_x_flat, _ = s_block(h_x_flat, h_v_flat, topo_bias)
            h_x_seq = h_x_flat.reshape(B, T, J, D)

            # Temporal: cross-frame joint attention
            h_x_seq = t_block(h_x_seq, h_v_seq)

        # Output head
        log_h = log_map0(h_x_seq.reshape(B * T * J, D))
        out = self.head(log_h[..., 1:])
        pred = out.reshape(B, T, J, 3)

        if return_manifold:
            return pred, h_x_seq
        return pred
