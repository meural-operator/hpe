import torch
import torch.nn as nn
from math_utils.lorentz import project

class PhaseSpaceEmbedding(nn.Module):
    def __init__(self, in_features, embed_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        """
        x: [B, N, in_features] or [B, T, N, in_features]
        Returns points in Lorentz manifold: [..., embed_dim + 1]
        """
        v = self.fc(x)
        
        # Add a dummy zero for the time coordinate (index 0)
        # project() will replace it with the correct sqrt(1 + sum(v_i^2))
        dummy = torch.zeros(*v.shape[:-1], 1, device=v.device, dtype=v.dtype)
        euclidean_x = torch.cat([dummy, v], dim=-1)
        
        return project(euclidean_x)
