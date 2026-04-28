"""
Riemannian Loss Functions for Hyperbolic HPE v2.

All losses operate natively on the Lorentz manifold — no Euclidean auxiliary patches.
Dynamic weighting via Kendall et al. (2018) uncertainty-based multi-task learning.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from math_utils.lorentz import dist, lorentz_inner, exp_map0, log_map0
from model.embedding import PhaseSpaceEmbedding


# H36M kinematic tree: (parent, child) edges
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),      # right leg
    (0, 4), (4, 5), (5, 6),      # left leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
]


def geodesic_velocity_loss(pred, gt):
    """
    Geodesic Velocity Consistency: measures temporal smoothness on the manifold.

    Instead of comparing Euclidean finite differences, we compare the geodesic
    distance between consecutive frames:
        L_vel = mean | d_L(pred_t, pred_{t+1}) - d_L(gt_t, gt_{t+1}) |

    This penalizes temporal jitter in a curvature-aware manner.

    Args:
        pred: [B, T, J, 3] predicted 3D poses (Euclidean)
        gt:   [B, T, J, 3] ground truth 3D poses (Euclidean)
    """
    B, T, J, C = pred.shape
    if T < 2:
        return torch.tensor(0.0, device=pred.device)

    # Embed into Lorentz for geodesic measurement
    # We pad with zero time-coord and project
    pred_flat = pred.view(B * T, J, C)
    gt_flat = gt.view(B * T, J, C)

    # Simple projection: pad with zeros and use project
    from math_utils.lorentz import project
    pred_h = project(torch.nn.functional.pad(pred_flat, (1, 0), value=0.0))  # [B*T, J, d+1]
    gt_h = project(torch.nn.functional.pad(gt_flat, (1, 0), value=0.0))

    pred_h = pred_h.view(B, T, J, -1)
    gt_h = gt_h.view(B, T, J, -1)

    # Geodesic distances between consecutive frames, per joint
    pred_dist = dist(pred_h[:, :-1], pred_h[:, 1:])  # [B, T-1, J]
    gt_dist = dist(gt_h[:, :-1], gt_h[:, 1:])        # [B, T-1, J]

    return torch.mean(torch.abs(pred_dist - gt_dist))


def geodesic_bone_loss(pred, gt):
    """
    Geodesic Bone Length Constraint: enforces skeletal rigidity using
    hyperbolic distances between parent-child joints.

        L_bone = mean | d_L(pred_i, pred_j) - d_L(gt_i, gt_j) |

    where (i, j) are edges in the H36M kinematic tree.

    Args:
        pred: [B, T, J, 3] predicted 3D poses
        gt:   [B, T, J, 3] ground truth 3D poses
    """
    B, T, J, C = pred.shape

    from math_utils.lorentz import project
    pred_flat = pred.view(B * T, J, C)
    gt_flat = gt.view(B * T, J, C)

    pred_h = project(torch.nn.functional.pad(pred_flat, (1, 0), value=0.0))  # [B*T, J, 4]
    gt_h   = project(torch.nn.functional.pad(gt_flat,   (1, 0), value=0.0))  # [B*T, J, 4]

    # Pre-built index tensors — one vectorized dist() call replaces 32 serial ones
    parents  = torch.tensor([p for p, _ in H36M_SKELETON], device=pred.device)  # [E]
    children = torch.tensor([c for _, c in H36M_SKELETON], device=pred.device)  # [E]

    pred_bone = dist(pred_h[:, parents], pred_h[:, children])  # [B*T, E]
    gt_bone   = dist(gt_h[:, parents],   gt_h[:, children])    # [B*T, E]

    return torch.mean(torch.abs(pred_bone - gt_bone))


def manifold_drift_loss(h_points):
    """
    Manifold Drift Regularizer: diagnostic metric measuring how far
    hidden representations deviate from the Lorentz constraint <x, x>_L = -1.

        L_drift = mean | <h_i, h_i>_L + 1 |

    If the geometry is healthy, this should be ~0. A rising drift indicates
    numerical instability in the Lorentz operations.

    Args:
        h_points: [B, N, d+1] points that should lie on H^d
    """
    sqnorm = lorentz_inner(h_points, h_points)  # should be -1.0
    return torch.mean(torch.abs(sqnorm + 1.0))


class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"

    Each loss L_i is weighted by a learnable log-variance parameter s_i:
        L_total = sum_i ( (1 / 2σ²_i) L_i + log(σ_i) )

    This naturally balances losses without manual tuning:
    - High-variance (noisy) losses get lower weight
    - Low-variance (reliable) losses get higher weight
    - The log(σ) term prevents all weights from collapsing to zero
    """

    def __init__(self, num_losses):
        super().__init__()
        # Initialize log-variances to 0 → σ = 1 → weight = 0.5
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *losses):
        """
        Args:
            losses: tuple of scalar loss tensors
        Returns:
            weighted_total: scalar
            weights: list of effective weights (1 / 2σ²) for logging
        """
        total = torch.tensor(0.0, device=losses[0].device)
        weights = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            # Detach the last loss (drift) — it's diagnostic, not a training signal.
            # The Lorentz exp_map guarantees manifold membership by construction;
            # drift only measures float32 round-off, not model failure.
            if i == len(losses) - 1:
                weights.append(precision.item())
                continue
            total = total + 0.5 * precision * loss + 0.5 * self.log_vars[i]
            weights.append(precision.item())
        return total, weights
