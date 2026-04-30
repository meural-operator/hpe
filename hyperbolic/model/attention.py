"""
Hyperbolic Attention modules — v3 (tangent-flow).

Key changes vs v2 (motivated by assessment §5–6):
  • Tangent-flow API: blocks accept and return tangent-at-origin representations
    of shape [..., d]. Manifold round-trips happen only inside HKPSA where Q,K
    must live on H^d for the geodesic logit, and only at the model boundary.
    → Eliminates ~6 log_o/exp_o pairs per forward pass.

  • Drop acosh from HKPSA logit. softmax is invariant under monotone transforms
    of logits, and -<q,k>_L = cosh(d_L(q,k)) is monotone in d_L. So we use the
    Lorentzian inner product directly: logit_geo = (1 + <q,k>_L) / tau, where
    the constant 1 keeps the score bounded above 0 at d=0 (-<x,x>_L = 1).
    → Saves an acosh, a clamp, and a square per pair, and removes a precision
       hot-spot in bf16.

  • Hierarchical kinematic topology bias: γ₁·A + γ₂·A² + γ₃·A³ captures parent /
    sibling / cousin relationships in the kinematic tree, not just adjacency.
    γ_k are learnable scalars.

  • Velocity is kept at origin (T_oH^d) throughout the network — the embedding
    no longer parallel-transports velocities to joint locations, eliminating
    the redundant per-block PT-back-to-origin.

  • Fused Q/K/V projection (one Linear(d, 3d)).

  • Multi-scale temporal windows configurable per block (W = 3, 9, 27 by default).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from math_utils.lorentz import exp_map0

HKPSA_TAN_BOUND = 3.0
# Why 3.0 and not the previous 15.0:
# At ‖v‖=15, cosh(15)≈1.6e6 and the Lorentzian inner product <q,k>_L between
# two such vectors lands near 1e12. Through `(1+<q,k>_L)/τ` the softmax sees
# logits with magnitude 1e12 → it saturates to one-hot regardless of which
# key is closer, so the spatial block can't actually distinguish joints.
# At ‖v‖=3, cosh(3)≈10, <q,k>_L is O(10²), and softmax has dynamic range to
# work with. Geodesic ordering is preserved (the bound is symmetric) and the
# fp32 overflow safety net at MAX_NORM=15 inside exp_map0 is no longer the
# binding constraint.


def _bound_tangent(v_tan, bound):
    """Bounds the Euclidean norm of spatial tangent vectors to prevent exp_o overflow."""
    norm = torch.sqrt(torch.sum(v_tan ** 2, dim=-1, keepdim=True).clamp_min(1e-7))
    scale = torch.clamp(bound / norm, max=1.0)
    return v_tan * scale


# Numerical operating principle for HKPSA
# ───────────────────────────────────────
# `exp_map0(v)` produces manifold coordinates of magnitude `cosh(‖v‖) ≈ e^‖v‖/2`
# and gradient magnitude `sinh(‖v‖) ≈ e^‖v‖/2`. We want both in O(1) so that
# (i) the Lorentzian inner product <q,k>_L = O(1), keeping softmax well-
# conditioned, (ii) drift due to fp32 catastrophic cancellation in
# −x_0² + ‖x_i‖² is negligible, and (iii) gradients to the QKV projections
# don't get amplified by exp(‖v‖).
#
# This is the hyperbolic analogue of scaled-dot-product attention's `/√d`:
# instead of dividing the *dot product* by √d to keep Euclidean logits in O(1),
# we divide the *tangent vectors* themselves by √d so the operating point of
# `exp_map0` stays near the origin where cosh ≈ 1 + ‖v‖²/2 (locally Euclidean).
#
# With this scaling, MAX_NORM=15 in math_utils.lorentz reverts to its proper
# role — a fp32 overflow safety net that should never be triggered in practice.


# ─────────────────────────────────────────────────────────────────────────────
# Spatial: Hyperbolic Kinematic Phase-Space Attention (tangent-flow)
# ─────────────────────────────────────────────────────────────────────────────
class HyperbolicKinematicAttention(nn.Module):
    """Spatial HKPSA — operates on tangent-at-origin inputs.

    Logits combine:
        s_geo  = (1 + <q,k>_L) / τ          (monotone-equivalent to -d_L²/τ
                                              after softmax; no acosh)
        s_kin  = -λ ‖v_i - v_j‖²            (velocity coherence, at origin)
        s_topo = γ₁ A + γ₂ A² + γ₃ A³       (hierarchical kinematic bias)

    Inputs
    ------
        x_tan  : [B, N, d]   tangent at origin (no time coord)
        v_tan  : [B, N, d]   velocity tangent at origin (no time coord)
        topo   : tuple/list of length K of [J, J] adjacency powers, OR a single
                 [J, J] matrix (then powers computed once and cached)

    Returns
    -------
        z_tan  : [B, N, d]   attended tangent at origin (no time coord)
    """

    def __init__(self, embed_dim, num_heads=8, lambda_penalty=1.0, num_topo_powers=3):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.lambda_penalty = lambda_penalty

        # Per-head learnable temperature — lets each head pick its own attention
        # sharpness. Same parameter count regime as a single scalar.
        self.tau = nn.Parameter(torch.ones(num_heads))

        # Fused Q/K/V projection — one matmul instead of three
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)

        # Per-head learnable hierarchical topology weights (γ_{h,k} for A^k).
        # Heads can specialise on different graph hops: e.g. one head locks onto
        # γ₁ (parent), another onto γ₂ (sibling/grandparent), etc.
        self.num_topo_powers = num_topo_powers
        self.topo_gamma = nn.Parameter(torch.zeros(num_heads, num_topo_powers))
        # Init γ₁=1, γ₂=γ₃=0 for every head — equivalent to uniform adjacency
        # bias at start, lets training differentiate heads.
        with torch.no_grad():
            self.topo_gamma[:, 0] = 1.0

        # Cached A^k powers, computed lazily on first forward (J×J small)
        self._A_powers_cache = None
        self._A_cache_id = None

    @staticmethod
    def _compute_A_powers(A: torch.Tensor, K: int) -> torch.Tensor:
        """Return [K, J, J] stack of A^1, A^2, ..., A^K, each row-normalised
        to keep magnitudes comparable across powers (deeper hops get diluted)."""
        powers = []
        cur = A
        for _ in range(K):
            # Clip to {0,1}: presence-of-path bias rather than count-of-paths,
            # then exclude self-loops introduced by even powers.
            mask = (cur > 0).float()
            mask = mask - torch.diag(torch.diag(mask))
            powers.append(mask)
            cur = cur @ A
        return torch.stack(powers, dim=0)        # [K, J, J]

    def forward(self, x_tan, v_tan, topo_bias=None):
        B, N, d = x_tan.shape
        H, d_h = self.num_heads, self.head_dim

        # ── Fused Q/K/V projection, split into heads ─────────────────────────
        # qkv: [B, N, 3d] → [B, N, 3, H, d_h]
        qkv = self.qkv_proj(x_tan).view(B, N, 3, H, d_h)
        # Bring head dim before token dim for batched matmuls: [B, H, N, d_h]
        q_s = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()
        k_s = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
        v_s = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

        # ── Geodesic-proximity logit (no acosh) ──────────────────────────────
        # Bound Q/K per-head: with H=8 and d=512, head dim is d_h=64 and the
        # per-head norm of a freshly-init Linear output is ≈ √d_h ≈ 8 — already
        # too large for cosh. Clip to HKPSA_TAN_BOUND so each head's softmax
        # stays in a usable dynamic range. Geodesic ordering is preserved.
        q_b = _bound_tangent(q_s, HKPSA_TAN_BOUND)                    # [B, H, N, d_h]
        k_b = _bound_tangent(k_s, HKPSA_TAN_BOUND)
        q = exp_map0(F.pad(q_b, (1, 0), value=0.0))                   # [B, H, N, d_h+1]
        k = exp_map0(F.pad(k_b, (1, 0), value=0.0))
        # <q_i, k_j>_L per head: split time/space and matmul over last dim
        time_term  = -q[..., 0:1] @ k[..., 0:1].transpose(-2, -1)
        space_term = q[..., 1:]   @ k[..., 1:].transpose(-2, -1)
        lor_inner  = time_term + space_term                            # [B, H, N, N]

        # Per-head temperature
        tau = torch.clamp(self.tau, min=1e-3).view(1, H, 1, 1)         # [1, H, 1, 1]
        s_geo = (1.0 + lor_inner) / tau                                # [B, H, N, N]

        # ── Kinematic velocity-coherence logit (per head) ────────────────────
        # v_tan reuses the position qkv projection — same parameter sharing
        # as the single-head version, just sliced per head.
        v_qkv = self.qkv_proj(v_tan).view(B, N, 3, H, d_h)
        v_q = v_qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()          # [B, H, N, d_h]
        v_k = v_qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
        vq_sq = (v_q ** 2).sum(-1, keepdim=True)                       # [B, H, N, 1]
        vk_sq = (v_k ** 2).sum(-1, keepdim=True)
        vqk   = v_q @ v_k.transpose(-2, -1)                            # [B, H, N, N]
        kin_penalty = (vq_sq + vk_sq.transpose(-2, -1) - 2.0 * vqk).clamp_min(0.0)
        s_kin = -self.lambda_penalty * kin_penalty                     # [B, H, N, N]

        # ── Hierarchical kinematic-tree bias (per head) ──────────────────────
        s_topo = 0.0
        if topo_bias is not None:
            if topo_bias.dim() == 2:
                cache_key = (id(topo_bias), tuple(topo_bias.shape), topo_bias.dtype)
                if self._A_cache_id != cache_key:
                    self._A_powers_cache = self._compute_A_powers(
                        topo_bias, self.num_topo_powers).to(topo_bias.dtype)
                    self._A_cache_id = cache_key
                A_pows = self._A_powers_cache                          # [K, J, J]
            else:
                A_pows = topo_bias                                      # [K, J, J] precomputed
            # γ_{h,k} · A^k → per-head bias [H, J, J]
            s_topo = torch.einsum('hk,kij->hij', self.topo_gamma, A_pows)
            s_topo = s_topo.unsqueeze(0)                                # [1, H, J, J]

        # ── Combined logits and softmax ──────────────────────────────────────
        logits = s_geo + s_kin + s_topo                                 # [B, H, N, N]
        attn_w = torch.softmax(logits, dim=-1)

        # ── Aggregation per head, then concat ────────────────────────────────
        z_per_head = attn_w @ v_s                                       # [B, H, N, d_h]
        # [B, H, N, d_h] → [B, N, H, d_h] → [B, N, d]
        z_tan = z_per_head.permute(0, 2, 1, 3).contiguous().view(B, N, d)
        return z_tan, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# Temporal: Windowed attention in tangent-at-origin space
# ─────────────────────────────────────────────────────────────────────────────
class HyperbolicTemporalAttention(nn.Module):
    """Per-joint windowed temporal attention.

    Operates entirely in T_o H^d (tangent at origin). Since the previous v2
    temporal block already did Euclidean dot-product attention in the tangent
    space (only log_o at start and exp_o at end were 'hyperbolic' operations),
    we drop those round-trips entirely — the temporal block is now a clean
    banded attention that costs O(T·W·d) per joint.

    Window W is a constructor argument so the network can stack different
    scales (e.g. 3, 9, 27) for multi-scale temporal context.
    """

    def __init__(self, embed_dim, temporal_window=3, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window = int(temporal_window)
        # Per-head temperature
        self.tau = nn.Parameter(torch.ones(num_heads))
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)

    def forward(self, x_tan_seq):
        """
        x_tan_seq: [B, T, J, d]   tangent at origin (no time coord)
        returns:   [B, T, J, d]   tangent at origin
        """
        B, T, J, d = x_tan_seq.shape
        W = self.window
        KW = 2 * W + 1
        H, d_h = self.num_heads, self.head_dim

        # Flatten joints into batch for efficient ops: [BJ, T, d]
        x = x_tan_seq.permute(0, 2, 1, 3).reshape(B * J, T, d)
        BJ = B * J

        # Project and split heads: [BJ, T, 3d] → [BJ, T, 3, H, d_h] → per-head tensors
        qkv = self.qkv_proj(x).view(BJ, T, 3, H, d_h)
        q_s = qkv[:, :, 0].permute(0, 2, 1, 3).contiguous()    # [BJ, H, T, d_h]
        k_s = qkv[:, :, 1].permute(0, 2, 1, 3).contiguous()
        v_s = qkv[:, :, 2].permute(0, 2, 1, 3).contiguous()

        # ── Build windowed K, V per head via unfold ──────────────────────────
        # Reshape to [BJ*H, d_h, T] for unfold over time, then re-split.
        k_flat = k_s.reshape(BJ * H, T, d_h).transpose(1, 2)   # [BJ*H, d_h, T]
        v_flat = v_s.reshape(BJ * H, T, d_h).transpose(1, 2)
        k_pad = F.pad(k_flat, (W, W), mode='replicate')         # [BJ*H, d_h, T+2W]
        v_pad = F.pad(v_flat, (W, W), mode='replicate')
        # unfold: [BJ*H, d_h, T, KW] → [BJ*H, T, KW, d_h]
        k_win = k_pad.unfold(2, KW, 1).permute(0, 2, 3, 1).contiguous()
        v_win = v_pad.unfold(2, KW, 1).permute(0, 2, 3, 1).contiguous()
        # Restore head dim: [BJ, H, T, KW, d_h]
        k_win = k_win.view(BJ, H, T, KW, d_h)
        v_win = v_win.view(BJ, H, T, KW, d_h)

        # ── Scaled dot-product within the window (per head) ──────────────────
        tau = torch.clamp(self.tau, min=1e-3).view(1, H, 1, 1)
        # logits[b,h,t,w] = (q_s[b,h,t] · k_win[b,h,t,w]) / (sqrt(d_h) · τ_h)
        logits = (q_s.unsqueeze(3) * k_win).sum(-1) / (math.sqrt(d_h) * tau)

        # ── Boundary mask ─────────────────────────────────────────────────────
        t_idx = torch.arange(T, device=x.device)
        w_idx = torch.arange(KW, device=x.device) - W
        frame_idx = t_idx.unsqueeze(1) + w_idx.unsqueeze(0)
        oob_mask = (frame_idx < 0) | (frame_idx >= T)               # [T, KW]
        # Broadcast over batch and head: [1, 1, T, KW]
        logits = logits.masked_fill(oob_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_w = torch.softmax(logits, dim=-1)                       # [BJ, H, T, KW]
        # [BJ, H, T, KW, 1] * [BJ, H, T, KW, d_h] → sum over KW → [BJ, H, T, d_h]
        agg = (attn_w.unsqueeze(-1) * v_win).sum(dim=3)
        # Concat heads: [BJ, H, T, d_h] → [BJ, T, H, d_h] → [BJ, T, d]
        agg = agg.permute(0, 2, 1, 3).contiguous().view(BJ, T, d)

        # Reshape back: [BJ, T, d] → [B, J, T, d] → [B, T, J, d]
        return agg.view(B, J, T, d).permute(0, 2, 1, 3)
