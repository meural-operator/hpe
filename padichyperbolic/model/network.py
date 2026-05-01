import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic.math_utils.lorentz import log_map0, exp_map0, origin, parallel_transport
from hyperbolic.model.embedding import PhaseSpaceEmbedding
from padichyperbolic.model.attention import HyperbolicKinematicAttention, PAdicTemporalBlock

class SpatialBlock(nn.Module):
    """Spatial attention block: joints attend to each other within a frame using Hyperbolic geometry."""

    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = HyperbolicKinematicAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, x_vel, topo_bias):
        """x: [B, N, d+1] on manifold"""
        z, attn_w, x_tan = self.attn(x, x_vel, topo_bias)
        z_tan = log_map0(z)[..., 1:]
        h = self.norm1(x_tan + self.drop(z_tan))
        
        ffn_out = self.fc2(self.drop(self.act(self.fc1(h))))
        h = self.norm2(h + self.drop(ffn_out))
        
        h_full = F.pad(h, (1, 0), value=0.0)
        return exp_map0(h_full), attn_w

class PAdicHyperbolicHPE(nn.Module):
    """
    P-Adic Hyperbolic Neural Operator for 3D HPE.
    Combines Lorentz manifold spatial constraints with P-Adic FNO temporal logic.
    """

    def __init__(self, in_features=3, embed_dim=512, num_spatial=3, num_temporal=3,
                 p=3, n=5, modes=15, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.embed = PhaseSpaceEmbedding(in_features, embed_dim)

        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(embed_dim, mlp_ratio, dropout)
            for _ in range(num_spatial)
        ])
        
        self.temporal_blocks = nn.ModuleList([
            PAdicTemporalBlock(embed_dim, p=p, n=n, modes=modes, mlp_ratio=mlp_ratio, dropout=dropout)
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

        # Interleaved Spatial (Hyperbolic) -> Temporal (P-Adic)
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: per-frame joint attention on Lorentz Manifold
            h_x_flat = h_x_seq.reshape(B * T, J, D)
            h_v_flat = h_v_seq.reshape(B * T, J, D)
            h_x_flat, _ = s_block(h_x_flat, h_v_flat, topo_bias)
            h_x_seq = h_x_flat.reshape(B, T, J, D)

            # Temporal: cross-frame joint attention using P-Adic FNO
            h_x_seq = t_block(h_x_seq, h_v_seq)

        # Output head
        log_h = log_map0(h_x_seq.reshape(B * T * J, D))
        out = self.head(log_h[..., 1:])
        pred = out.reshape(B, T, J, 3)

        if return_manifold:
            return pred, h_x_seq
        return pred
