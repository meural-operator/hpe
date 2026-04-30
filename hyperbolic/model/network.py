"""
HyperbolicHPE network — v3.

Architectural changes (vs v2):

1. Tangent-flow data path
   Hidden state is carried in T_o H^d (no time coord) shape [B, T, J, d].
   Manifold representations are constructed only inside HKPSA (where Q,K
   need to be on the hyperboloid for the geodesic logit) and at the optional
   `return_manifold` exit (one exp_o on the final tangent state for the
   drift diagnostic). This eliminates ~6 log_o / exp_o pairs per forward.

2. Multi-scale temporal windows
   The 3 temporal blocks now use windows W ∈ {3, 9, 27} instead of all W=3.
   Effective receptive field: 3·(2W+1)−2  ≈ 75 frames covering H36M's gait
   cycle, vs ~19 frames before. Configurable via `temporal_windows`.

3. Velocity kept at origin
   The phase-space embedding no longer parallel-transports velocities to
   joint locations (and the spatial block no longer transports them back).
   Velocities live in T_o H^d throughout, removing 2 PT calls per spatial
   block. Mathematically: HKPSA already compares velocities AT the origin,
   so PT-out-then-PT-back was always a no-op modulo numerical drift.

4. Per-joint output head
   Replaces the shared `Linear(d, 3)` decoder with 17 small per-joint heads,
   decoupling joints at decode time. Tiny param cost, often worth ~0.5–1 mm.

5. Hierarchical kinematic bias
   Topology bias is now γ₁A + γ₂A² + γ₃A³ (parent / sibling / cousin) with
   learnable γ_k. Adjacency powers cached once per device.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import HyperbolicKinematicAttention, HyperbolicTemporalAttention
from model.embedding import PhaseSpaceEmbedding
from math_utils.lorentz import exp_map0, log_map0


# ─────────────────────────────────────────────────────────────────────────────
# FFN in the tangent space at origin
# ─────────────────────────────────────────────────────────────────────────────
class TangentFFN(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# Spatial block (tangent-in, tangent-out)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialBlock(nn.Module):
    """Spatial reasoning: joints attend to each other within a single frame.

    Pre-norm transformer style: LayerNorm is applied BEFORE attention and FFN
    so that the inputs to HKPSA's `exp_map0(F.pad(q_s, ...))` have bounded
    magnitude. Without pre-norm, raw tangent vectors with norm ≈ √d ≈ 22.6
    exceed `MAX_NORM=15` inside `exp_map0`, sending Q/K off-manifold and
    causing gradient blow-up.
    """

    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = HyperbolicKinematicAttention(embed_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)            # normalize velocity too
        self.ffn = TangentFFN(embed_dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_tan, v_tan, topo_bias):
        """
        x_tan, v_tan : [B, J, d]   tangent-at-origin (no time coord)
        topo_bias    : [J, J]      adjacency (powers computed inside attention)
        """
        # Pre-norm: normalize tangent representations before they hit HKPSA.
        z_tan, _ = self.attn(self.norm1(x_tan), self.norm_v(v_tan), topo_bias)
        h = x_tan + self.drop(z_tan)
        h = h + self.ffn(self.norm2(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Temporal block (tangent-in, tangent-out)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalBlock(nn.Module):
    """Cross-frame reasoning: same joint attends across a temporal window."""

    def __init__(self, embed_dim, temporal_window=3, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = HyperbolicTemporalAttention(embed_dim, temporal_window, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = TangentFFN(embed_dim, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_tan_seq):
        """x_tan_seq: [B, T, J, d]"""
        # Pre-norm: keeps temporal-attention QK products in a stable regime.
        z = self.attn(self.norm1(x_tan_seq))
        h = x_tan_seq + self.drop(z)
        h = h + self.ffn(self.norm2(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Per-joint output head
# ─────────────────────────────────────────────────────────────────────────────
class PerJointHead(nn.Module):
    """Decodes each joint independently from its d-dim tangent representation.

    Implemented as a single fused linear over a per-joint expert bank to keep
    GEMM efficiency: weights have shape [J, d, 3], biases [J, 3].
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
        # Init like nn.Linear: kaiming-uniform with gain for relu/gelu
        nn.init.kaiming_uniform_(self.fc1_w, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.fc2_w, a=5 ** 0.5)

    def forward(self, x):
        """x: [B, T, J, d]  → [B, T, J, 3]"""
        x = self.norm(x)
        # einsum: per-joint linear
        h = torch.einsum('btjd,jdh->btjh', x, self.fc1_w) + self.fc1_b
        h = self.act(h)
        y = torch.einsum('btjh,jho->btjo', h, self.fc2_w) + self.fc2_b
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Main network
# ─────────────────────────────────────────────────────────────────────────────
class HyperbolicHPE(nn.Module):
    """Hyperbolic Kinematic Phase-Space Attention Network — v3.

    Pipeline
    --------
        2D keypoints  →  PhaseSpaceEmbedding (tangent at origin)
                      →  L × (Spatial → Temporal_W_l)
                      →  PerJointHead  →  3D coordinates

    All blocks operate in T_o H^d. The optional `return_manifold` exits the
    final tangent state through exp_o for drift diagnostics.
    """

    def __init__(
        self,
        in_features=3,
        embed_dim=512,
        num_spatial=3,
        num_temporal=3,
        num_heads=8,
        temporal_window=3,                     # backward-compat: scalar or list
        temporal_windows=None,                 # preferred: list of length num_temporal
        mlp_ratio=4,
        dropout=0.1,
        num_joints=17,
    ):
        super().__init__()

        # Resolve temporal windows: prefer multi-scale list, fall back to scalar.
        if temporal_windows is None:
            if isinstance(temporal_window, (list, tuple)):
                temporal_windows = list(temporal_window)
            else:
                # Default multi-scale schedule when only a scalar is given but
                # num_temporal == 3: covers short / medium / long temporal context.
                if num_temporal == 3:
                    temporal_windows = [3, 9, 27]
                else:
                    temporal_windows = [int(temporal_window)] * num_temporal
        assert len(temporal_windows) == num_temporal, (
            f"temporal_windows must have length num_temporal={num_temporal}, "
            f"got {temporal_windows}"
        )
        self.temporal_windows = temporal_windows

        self.embed = PhaseSpaceEmbedding(in_features, embed_dim)

        # Joint-id embedding: gives each joint a learned d-dim signature added
        # to its tangent representation before the first spatial block. Without
        # this the only joint-distinguishing signal in the input is the topology
        # bias inside HKPSA — and HKPSA only sees pairs (i, j), not joint
        # identity per token. Standard trick in pose lifters; ~17·d params.
        self.num_joints = num_joints
        self.joint_embed = nn.Parameter(torch.zeros(num_joints, embed_dim))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        self.spatial_blocks = nn.ModuleList([
            SpatialBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_spatial)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(embed_dim, temporal_window=W, num_heads=num_heads,
                          mlp_ratio=mlp_ratio, dropout=dropout)
            for W in temporal_windows
        ])

        self.head = PerJointHead(embed_dim, num_joints)

    # ── Embedding helpers ────────────────────────────────────────────────────
    def _embed_position_to_tangent(self, x):
        """Position embedding → tangent at origin.

        Matches v2's numerical regime: linear → project(pad0(·)) onto H^d →
        log_map0 → drop time coord. The compounding `arcsinh(‖Wx+b‖)`
        non-linearity bounds initial tangent norms to ≤ MAX_NORM=15, which is
        what HKPSA's `exp_map0` needs to keep Q/K on the hyperboloid.
        """
        h_manifold = self.embed(x)                 # [..., d+1] on H^d
        return log_map0(h_manifold)[..., 1:]       # [..., d] tangent at origin

    def _embed_velocity_to_tangent(self, x_vel):
        """Velocity embedding → tangent at origin.

        Velocities are tangent vectors by construction (finite differences of
        points on H^d). Linear projection with no manifold round-trip;
        confidence delta is dropped inside `embed.velocity` since it carries
        no useful signal.
        """
        return self.embed.velocity(x_vel)

    def forward(self, x, x_vel, topo_bias=None, return_manifold=False):
        """
        x       : [B, T, J, in_features]  — 2D keypoints (with confidence)
        x_vel   : [B, T, J, in_features]  — finite-difference velocities
        topo_bias : [J, J]                — kinematic-tree adjacency
        """
        B, T, J, _ = x.shape

        # ── Embed positions and velocities to tangent at origin ───────────────
        h_x = self._embed_position_to_tangent(x)        # [B, T, J, d]
        h_v = self._embed_velocity_to_tangent(x_vel)    # [B, T, J, d] — kept at origin

        # Add per-joint identity signature (broadcast over [B, T])
        h_x = h_x + self.joint_embed.view(1, 1, J, -1)

        d = h_x.shape[-1]

        # ── Interleaved Spatial → Temporal blocks ─────────────────────────────
        for s_block, t_block in zip(self.spatial_blocks, self.temporal_blocks):
            # Spatial: per-frame joint attention — flatten T into batch.
            h_x_flat = h_x.reshape(B * T, J, d)
            v_flat   = h_v.reshape(B * T, J, d)
            h_x_flat = s_block(h_x_flat, v_flat, topo_bias)
            h_x = h_x_flat.reshape(B, T, J, d)

            # Temporal: per-joint cross-frame attention.
            h_x = t_block(h_x)

        # ── Decode ────────────────────────────────────────────────────────────
        pred = self.head(h_x)                       # [B, T, J, 3]

        if return_manifold:
            # Map the final tangent state to the manifold for the drift
            # diagnostic. Bound tangent norms first — without it, residual
            # accumulation through 3 blocks can push norms past MAX_NORM=15
            # and the clamped exp_o output sits off the hyperboloid.
            h_flat = h_x.reshape(-1, d)
            n = torch.sqrt(h_flat.pow(2).sum(-1, keepdim=True).clamp_min(1e-7))
            h_flat = h_flat * torch.clamp(5.0 / n, max=1.0)
            h_full = F.pad(h_flat, (1, 0), value=0.0)
            h_manifold = exp_map0(h_full).reshape(B, T, J, d + 1)
            return pred, h_manifold
        return pred
