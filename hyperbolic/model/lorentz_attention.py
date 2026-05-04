"""
Fully Hyperbolic Attention modules — v4.

Two attention mechanisms operating directly on the Lorentz manifold:

1. LorentzKinematicAttention (Spatial):
   - Q/K/V via LorentzLinear per-head (full boost + rotation)
   - Geodesic logit + kinematic velocity penalty + topology bias
   - On-manifold aggregation via spatial weighted sum + re-projection

2. LorentzLinearTemporalAttention (Temporal):
   - O(T) linear attention using positive feature map
   - Q/K/V via LorentzLinear
   - Global receptive field — no windowing
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from math_utils.lorentz import project, EPS
from model.lorentz_layers import LorentzLinear


# ─────────────────────────────────────────────────────────────────────────────
# Spatial: Lorentz Kinematic Attention (fully on-manifold)
# ─────────────────────────────────────────────────────────────────────────────
class LorentzKinematicAttention(nn.Module):
    """Spatial attention on the Lorentz manifold.

    Q, K, V are produced by LorentzLinear (full Lorentz transformation).
    Logits combine geodesic proximity, velocity coherence, and topology bias.
    Aggregation: weighted sum of V's spatial components + re-projection.
    """

    def __init__(self, embed_dim, num_heads=8, lambda_penalty=1.0,
                 num_topo_powers=3, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim      # spatial dim d
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_h per head
        self.lambda_penalty = lambda_penalty

        # Per-head learnable temperature
        self.tau = nn.Parameter(torch.ones(num_heads))

        # Q/K/V: one fused linear on spatial components, then split & project
        # Input spatial d → output 3*d (will be split into Q, K, V)
        self.qkv_spatial = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        nn.init.xavier_uniform_(self.qkv_spatial.weight, gain=0.02)
        nn.init.zeros_(self.qkv_spatial.bias)

        # Velocity projection (stays in tangent/Euclidean space)
        self.vel_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection (on manifold)
        self.out_proj = LorentzLinear(embed_dim, embed_dim, dropout=0.0)

        # Hierarchical topology bias
        self.num_topo_powers = num_topo_powers
        self.topo_gamma = nn.Parameter(torch.zeros(num_heads, num_topo_powers))
        with torch.no_grad():
            self.topo_gamma[:, 0] = 1.0
        self._A_powers_cache = None
        self._A_cache_id = None

    @staticmethod
    def _compute_A_powers(A, K):
        powers = []
        cur = A
        for _ in range(K):
            mask = (cur > 0).float()
            mask = mask - torch.diag(torch.diag(mask))
            powers.append(mask)
            cur = cur @ A
        return torch.stack(powers, dim=0)

    def forward(self, x, v_tan, topo_bias=None):
        """
        Args:
            x:      [B, N, d+1]  points on H^d (Lorentz vectors)
            v_tan:  [B, N, d]    velocity tangent vectors (Euclidean)
            topo_bias: [J, J]    kinematic adjacency matrix
        Returns:
            z:      [B, N, d+1]  output on H^d
            attn_w: [B, H, N, N] attention weights
        """
        B, N, D = x.shape
        d = D - 1  # spatial dim
        H = self.num_heads
        d_h = self.head_dim

        # ── Q/K/V: project spatial components, split into heads ──────────────
        x_spatial = x[..., 1:]  # [B, N, d]
        qkv = self.qkv_spatial(x_spatial)  # [B, N, 3*d]
        qkv = qkv.view(B, N, 3, H, d_h)
        q_s = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [B, H, N, d_h]
        k_s = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
        v_s = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

        # Project each to hyperboloid per-head: t = sqrt(1 + ||s||²)
        def _to_lorentz(s):
            t = torch.sqrt(1.0 + (s * s).sum(-1, keepdim=True).clamp(min=EPS))
            return torch.cat([t, s], dim=-1)  # [..., d_h+1]

        q = _to_lorentz(q_s)  # [B, H, N, d_h+1]
        k = _to_lorentz(k_s)

        # ── Geodesic logit: (1 + ⟨q,k⟩_L) / τ ──────────────────────────────
        # ⟨q,k⟩_L pairwise = -q₀·k₀ᵀ + q_{1:}·k_{1:}ᵀ
        time_term = -(q[..., :1]) @ (k[..., :1]).transpose(-2, -1)   # [B,H,N,N]
        space_term = q[..., 1:] @ k[..., 1:].transpose(-2, -1)       # [B,H,N,N]
        lor_inner = time_term + space_term

        tau = torch.clamp(self.tau, min=1e-3).view(1, H, 1, 1)
        s_geo = (1.0 + lor_inner) / tau  # [B, H, N, N]

        # ── Velocity coherence logit ─────────────────────────────────────────
        v_proj = self.vel_proj(v_tan).view(B, N, H, d_h)
        v_proj = v_proj.permute(0, 2, 1, 3).contiguous()  # [B, H, N, d_h]
        vq_sq = (v_proj ** 2).sum(-1, keepdim=True)         # [B, H, N, 1]
        vqk = v_proj @ v_proj.transpose(-2, -1)              # [B, H, N, N]
        kin_penalty = (vq_sq + vq_sq.transpose(-2, -1) - 2.0 * vqk).clamp_min(0)
        s_kin = -self.lambda_penalty * kin_penalty

        # ── Hierarchical topology bias ───────────────────────────────────────
        s_topo = 0.0
        if topo_bias is not None:
            if topo_bias.dim() == 2:
                cache_key = (id(topo_bias), tuple(topo_bias.shape), topo_bias.dtype)
                if self._A_cache_id != cache_key:
                    self._A_powers_cache = self._compute_A_powers(
                        topo_bias, self.num_topo_powers).to(topo_bias.dtype)
                    self._A_cache_id = cache_key
                A_pows = self._A_powers_cache
            else:
                A_pows = topo_bias
            s_topo = torch.einsum('hk,kij->hij', self.topo_gamma, A_pows)
            s_topo = s_topo.unsqueeze(0)

        # ── Softmax → attention weights ──────────────────────────────────────
        logits = s_geo + s_kin + s_topo
        attn_w = torch.softmax(logits, dim=-1)  # [B, H, N, N]

        # ── Aggregate V spatial components, re-project to H^d ────────────────
        agg_spatial = attn_w @ v_s  # [B, H, N, d_h]

        # Merge heads: [B, H, N, d_h] → [B, N, H*d_h] = [B, N, d]
        agg_flat = agg_spatial.permute(0, 2, 1, 3).contiguous().view(B, N, d)

        # Re-project to hyperboloid
        agg_time = torch.sqrt(
            1.0 + (agg_flat * agg_flat).sum(-1, keepdim=True).clamp(min=EPS)
        )
        z = torch.cat([agg_time, agg_flat], dim=-1)  # [B, N, d+1]

        # Output projection (LorentzLinear)
        z = self.out_proj(z)
        return z, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# Temporal: Lorentz Linear Attention — O(T) global
# ─────────────────────────────────────────────────────────────────────────────
class LorentzLinearTemporalAttention(nn.Module):
    """O(T) linear temporal attention on the Lorentz manifold.

    Instead of computing the full T×T attention matrix (O(T²)), we use
    the linear attention trick (Katharopoulos et al. 2020):
        Attn(Q,K,V) = φ(Q) · (φ(K)ᵀ · V)  instead of  softmax(QKᵀ) · V
    where φ(x) = elu(x) + 1 is a positive feature map.

    This gives O(T·d²) complexity instead of O(T²·d), and provides
    GLOBAL temporal receptive field (every frame sees every other frame).
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q/K/V projection on spatial components
        self.qkv_spatial = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        nn.init.xavier_uniform_(self.qkv_spatial.weight, gain=0.02)
        nn.init.zeros_(self.qkv_spatial.bias)

        # Output projection
        self.out_proj = LorentzLinear(embed_dim, embed_dim, dropout=0.0)

    @staticmethod
    def _feature_map(x):
        """Positive feature map: φ(x) = elu(x) + 1."""
        return F.elu(x) + 1.0

    def forward(self, x):
        """
        Args:
            x: [B, T, J, d+1] points on H^d
        Returns:
            z: [B, T, J, d+1] output on H^d
        """
        B, T, J, D = x.shape
        d = D - 1
        H = self.num_heads
        d_h = self.head_dim

        # Flatten joints into batch: [B*J, T, d+1]
        x_flat = x.permute(0, 2, 1, 3).reshape(B * J, T, D)
        BJ = B * J

        # ── Q/K/V from spatial components ────────────────────────────────────
        x_spatial = x_flat[..., 1:]  # [BJ, T, d]
        qkv = self.qkv_spatial(x_spatial).view(BJ, T, 3, H, d_h)
        q = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()  # [BJ, H, T, d_h]
        k = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
        v = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

        # ── Linear attention: φ(Q) · (φ(K)ᵀ · V) ───────────────────────────
        q_mapped = self._feature_map(q)  # [BJ, H, T, d_h]
        k_mapped = self._feature_map(k)  # [BJ, H, T, d_h]

        # Compute KV: [BJ, H, d_h, d_h] = K^T @ V
        kv = k_mapped.transpose(-2, -1) @ v  # [BJ, H, d_h, d_h]

        # Compute output: Q @ KV = [BJ, H, T, d_h]
        agg = q_mapped @ kv  # [BJ, H, T, d_h]

        # Normalize by sum of attention weights (for stability)
        z_denom = (q_mapped @ k_mapped.transpose(-2, -1).sum(-1, keepdim=True)
                   ).clamp(min=EPS)
        # Simpler normalization: sum of φ(K) per query
        k_sum = k_mapped.sum(dim=-2, keepdim=True)  # [BJ, H, 1, d_h]
        z_denom = (q_mapped * k_sum).sum(-1, keepdim=True).clamp(min=EPS)
        agg = agg / z_denom  # [BJ, H, T, d_h]

        # ── Merge heads: [BJ, H, T, d_h] → [BJ, T, d] ──────────────────────
        agg_flat = agg.permute(0, 2, 1, 3).contiguous().view(BJ, T, d)

        # Re-project to hyperboloid
        agg_time = torch.sqrt(
            1.0 + (agg_flat * agg_flat).sum(-1, keepdim=True).clamp(min=EPS)
        )
        z = torch.cat([agg_time, agg_flat], dim=-1)  # [BJ, T, d+1]

        # Output projection
        z = self.out_proj(z)

        # Reshape back: [BJ, T, d+1] → [B, J, T, d+1] → [B, T, J, d+1]
        z = z.view(B, J, T, D).permute(0, 2, 1, 3)
        return z
