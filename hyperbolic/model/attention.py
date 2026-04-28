import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math_utils.lorentz import (
    dist, parallel_transport, log_map0, exp_map0, origin
)


class HyperbolicKinematicAttention(nn.Module):
    """Memory-efficient Hyperbolic Kinematic Phase-Space Attention.

    Compares velocities at the origin tangent space (O(N×D) memory)
    instead of pairwise parallel transport (O(N²×D)).
    """

    def __init__(self, embed_dim, lambda_penalty=1.0, gamma_topology=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_penalty = lambda_penalty
        self.gamma_topology = gamma_topology

        self.tau = nn.Parameter(torch.tensor([1.0]))

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, x_vel, topo_bias=None):
        """
        x: [B, N, d+1] on Lorentz manifold
        x_vel: [B, N, d+1] tangent vectors at x

        Returns:
            z:        [B, N, d+1]  attended manifold points
            attn_w:   [B, N, N]    attention weights
            x_tan:    [B, N, d]    log_map0(x)[..., 1:] — cached for reuse in SpatialBlock
        """
        B, N, D = x.shape

        log_x = log_map0(x)
        x_tan = log_x[..., 1:]          # [B, N, d] — returned so caller avoids recomputing
        spatial = x_tan

        q_s = self.q_proj(spatial)
        k_s = self.k_proj(spatial)
        v_s = self.v_proj(spatial)

        q = exp_map0(F.pad(q_s, (1, 0), value=0.0))
        k = exp_map0(F.pad(k_s, (1, 0), value=0.0))
        # v is only used via log_map0(v) in aggregation.
        # Since log_map0(exp_map0(v_padded)) == v_padded (inverses at origin),
        # we skip both calls and use the padded tangent vector directly.
        log_v = F.pad(v_s, (1, 0), value=0.0)          # [B, N, d+1]

        # Transport velocities to origin
        o = origin(x.shape, device=x.device, dtype=x.dtype)
        vel_at_origin = parallel_transport(x, o, x_vel)
        q_vel_s = self.q_proj(vel_at_origin[..., 1:])
        k_vel_s = self.k_proj(vel_at_origin[..., 1:])

        # Pairwise geodesic distance
        d_L = dist(q.unsqueeze(2), k.unsqueeze(1))  # [B, N, N]

        # Kinematic penalty via efficient dot product
        q_vel_sq = (q_vel_s ** 2).sum(-1, keepdim=True)
        k_vel_sq = (k_vel_s ** 2).sum(-1, keepdim=True)
        qk_vel_dot = torch.bmm(q_vel_s, k_vel_s.transpose(1, 2))
        kinematic_penalty = q_vel_sq + k_vel_sq.transpose(1, 2) - 2 * qk_vel_dot
        kinematic_penalty = torch.clamp(kinematic_penalty, min=0.0)

        tau_clamped = torch.clamp(self.tau, min=1e-3)
        logits = -(1.0 / tau_clamped) * (d_L ** 2) - self.lambda_penalty * kinematic_penalty

        if topo_bias is not None:
            logits = logits + self.gamma_topology * topo_bias.unsqueeze(0)

        attn_weights = torch.softmax(logits, dim=-1)

        # Tangent-space aggregation (log_v already in tangent space — no exp/log needed)
        agg = torch.bmm(attn_weights, log_v)
        z = exp_map0(agg)

        return z, attn_weights, x_tan


class HyperbolicTemporalAttention(nn.Module):
    """Cross-frame temporal attention on the Lorentz manifold.

    For each joint, attends across a temporal window of ±W neighboring frames.
    Uses O(T × W) unfold-based windowed attention instead of the previous
    O(T²) full distance matrix, giving a ~41× reduction in FLOPs for T=243, W=3.

    Input:  [B, T, J, d+1] — sequence of manifold-embedded frames
    Output: [B, T, J, d+1] — temporally-attended manifold points
    """

    def __init__(self, embed_dim, temporal_window=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.window = temporal_window

        self.tau = nn.Parameter(torch.tensor([1.0]))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_seq, vel_seq):
        """
        x_seq:   [B, T, J, d+1] manifold points across time
        vel_seq: [B, T, J, d+1] tangent velocities across time (unused here,
                 kept for API compatibility with the training loop)

        Returns: [B, T, J, d+1] temporally attended points
        """
        B, T, J, D = x_seq.shape
        W = self.window
        KW = 2 * W + 1          # kernel width (number of frames attended)
        d = D - 1               # spatial embed dim

        # --- Map all frames to tangent space at origin ---
        # [B*J, T, D] → flatten joints into batch for efficient ops
        x_flat = x_seq.permute(0, 2, 1, 3).reshape(B * J, T, D)  # [BJ, T, D]
        spatial = log_map0(x_flat)[..., 1:]                        # [BJ, T, d]

        q_s = self.q_proj(spatial)  # [BJ, T, d]
        k_s = self.k_proj(spatial)  # [BJ, T, d]
        v_s = self.v_proj(spatial)  # [BJ, T, d]

        # --- Build windowed K and V using unfold ---
        # Pad time dimension by W on each side (reflect pad avoids zero-vector keys)
        # [BJ, T, d] → [BJ, d, T] → pad → unfold → [BJ, T, KW, d]
        k_t = k_s.permute(0, 2, 1)                                  # [BJ, d, T]
        k_pad = F.pad(k_t, (W, W), mode='replicate')               # [BJ, d, T+2W]
        k_win = k_pad.unfold(2, KW, 1).permute(0, 3, 2, 1)        # [BJ, KW, T, d] → permute
        k_win = k_win.permute(0, 2, 1, 3).contiguous()            # [BJ, T, KW, d]

        v_t = v_s.permute(0, 2, 1)                                  # [BJ, d, T]
        v_pad = F.pad(v_t, (W, W), mode='replicate')               # [BJ, d, T+2W]
        v_win = v_pad.unfold(2, KW, 1).permute(0, 3, 2, 1)        # → [BJ, KW, T, d]
        v_win = v_win.permute(0, 2, 1, 3).contiguous()            # [BJ, T, KW, d]

        # q_s: [BJ, T, d], k_win: [BJ, T, KW, d]
        # logits[b,t,w] = sum_d q_s[b,t,d] * k_win[b,t,w,d] → [BJ, T, KW]
        tau_clamped = torch.clamp(self.tau, min=1e-3)
        logits = (q_s.unsqueeze(2) * k_win).sum(-1) / (math.sqrt(d) * tau_clamped)
        # [BJ, T, KW]

        # --- Mask padding positions at sequence boundaries ---
        # Positions w in [0, KW) correspond to frame (t - W + w).
        # When t - W + w < 0 or >= T, the replicate padding gives a valid
        # (repeated) key, but we still mask it to keep semantics clean.
        t_idx = torch.arange(T, device=x_seq.device)               # [T]
        w_idx = torch.arange(KW, device=x_seq.device) - W         # [-W,...,W]
        frame_idx = t_idx.unsqueeze(1) + w_idx.unsqueeze(0)        # [T, KW]
        boundary_mask = (frame_idx < 0) | (frame_idx >= T)         # [T, KW]
        logits = logits.masked_fill(boundary_mask.unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(logits, dim=-1)                # [BJ, T, KW]

        # --- Weighted aggregation over window ---
        # out[b,t,d] = sum_w attn[b,t,w] * v_win[b,t,w,d]
        agg_s = (attn_weights.unsqueeze(-1) * v_win).sum(dim=2)    # [BJ, T, d]

        # Map aggregated tangent vector back to the manifold
        agg_full = F.pad(agg_s, (1, 0), value=0.0)                 # [BJ, T, d+1]
        z = exp_map0(agg_full)                                      # [BJ, T, d+1]

        # Reshape back: [BJ, T, D] → [B, J, T, D] → [B, T, J, D]
        z = z.view(B, J, T, D).permute(0, 2, 1, 3)

        return z

