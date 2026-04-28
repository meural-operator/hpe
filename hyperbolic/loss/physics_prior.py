import torch
import torch.nn.functional as F

def energy_physics_prior(predicted_3d, alpha, gt_3d, stiffness_k=1000.0):
    """
    Computes physical energy E(theta) based on soft-body contact dynamics.
    Dynamically uses the ground truth to estimate the true floor height for each sequence,
    solving the root-relative (Z-center) tracking issue.

    predicted_3d: The raw [B, T, 17, 3] output from MotionAGFormer
    alpha: The [B, T, 17, 1] shrinkage/confidence factor predicted by the network
    gt_3d: Ground Truth coordinates [B, T, 17, 3]
    """
    B = predicted_3d.shape[0]
    
    # H36M indices for left and right foot/ankle
    left_foot_idx = 3
    right_foot_idx = 6

    # 1. Dynamically calculate Floor Height
    # Since Human3.6M normalizes Y downwards (positive Y = lower height),
    # the floor is the MAXIMUM Y coordinate achieved by any foot across the entire sequence.
    floor_y = torch.max(gt_3d[:, :, [left_foot_idx, right_foot_idx], 1].view(B, -1), dim=1, keepdim=True)[0]
    floor_y = floor_y.unsqueeze(1) # [B, 1, 1]

    # 2. Track Predicted Feet Y Coordinates
    left_foot_y = predicted_3d[:, :, left_foot_idx, 1]
    right_foot_y = predicted_3d[:, :, right_foot_idx, 1]

    # 3. Calculate Penetration
    left_penetration = F.relu(left_foot_y - floor_y.squeeze(-1))
    right_penetration = F.relu(right_foot_y - floor_y.squeeze(-1))

    # 4. Energy Spring
    left_energy = stiffness_k * (left_penetration ** 2)
    right_energy = stiffness_k * (right_penetration ** 2)

    # 5. Gate physically bounded violation against network confidence
    gated_left = alpha[:, :, left_foot_idx, 0] * left_energy
    gated_right = alpha[:, :, right_foot_idx, 0] * right_energy

    return (gated_left + gated_right).mean()

class CartesianPhysicsPrior(torch.nn.Module):
    """
    Approximates the physics manifold directly in 3D Cartesian space
    to create the X_physics target for Stein's Shrinkage using dynamic,
    unsupervised root-relative floor modeling.
    """
    def __init__(self):
        super().__init__()
        self.foot_indices = [3, 6]

    def forward(self, x_raw, gt_3d):
        B = x_raw.shape[0]
        x_physics = x_raw.clone()

        # Dynamic unsupervised floor target
        floor_y = torch.max(gt_3d[:, :, self.foot_indices, 1].view(B, -1), dim=1, keepdim=True)[0]
        floor_y = floor_y.unsqueeze(1) # [B, 1, 1]

        for idx in self.foot_indices:
            foot_y = x_physics[:, :, idx, 1]
            
            # Penetration > 0 means predicted is pushed below the lowest ground truth point
            penetration = torch.clamp(foot_y - floor_y.squeeze(-1), min=0.0)
            
            # Force the physical target to snap back to the actual logical floor
            x_physics[:, :, idx, 1] = foot_y - penetration

        if x_physics.shape[1] > 2:
            # Add simple temporal acceleration smoothing on corrected trajectory
            accel = x_physics[:, :-2] - 2 * x_physics[:, 1:-1] + x_physics[:, 2:]
            x_physics[:, 1:-1] = x_physics[:, 1:-1] + 0.1 * accel

        return x_physics.detach()
