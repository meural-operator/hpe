import torch
import torch.nn as nn
from math_utils.lorentz import project


class PhaseSpaceEmbedding(nn.Module):
    """Confidence-gated phase-space embedding.

    Inputs are CPN keypoints `(x, y, conf)`. The position embedding is gated
    by the confidence channel via a learned scalar gain in (0, 2):

        gate(c) = 1 + tanh(γ·c + β)

    Low-confidence joints (e.g. occluded ones) get attenuated representations
    so the network can route around them; high-confidence joints get
    amplified. γ and β are learned, init to (1, 0) so the first forward is
    near-identity (gate ≈ 1 + tanh(c) ≈ 1.76 at c=1).

    Position and velocity get independent linear maps. The confidence
    channel of `x_vel` (a finite difference of confidences) is meaningless
    and is dropped before projection.
    """

    def __init__(self, in_features=3, embed_dim=512):
        super().__init__()
        assert in_features == 3, (
            "PhaseSpaceEmbedding expects (x, y, conf); got in_features="
            f"{in_features}"
        )
        self.embed_dim = embed_dim
        self.embed_pos = nn.Linear(2, embed_dim)
        self.embed_vel = nn.Linear(2, embed_dim)
        self.conf_gain = nn.Parameter(torch.tensor(1.0))
        self.conf_bias = nn.Parameter(torch.tensor(0.0))

    def position(self, x):
        """x: [..., 3] = (x, y, conf) → manifold point on H^d, [..., d+1]."""
        xy, conf = x[..., :2], x[..., 2:3]
        v = self.embed_pos(xy)
        gate = 1.0 + torch.tanh(self.conf_gain * conf + self.conf_bias)
        v = v * gate
        dummy = torch.zeros(*v.shape[:-1], 1, device=v.device, dtype=v.dtype)
        return project(torch.cat([dummy, v], dim=-1))

    def velocity(self, x_vel):
        """x_vel: [..., 3] = (Δx, Δy, Δconf) → tangent at origin, [..., d].

        We drop Δconf — it carries no useful signal (it's the difference of
        two confidence scalars, which were either CPN softmax outputs or 1.0
        fallbacks, so its temporal derivative is essentially noise).
        """
        return self.embed_vel(x_vel[..., :2])

    def forward(self, x):
        # Backward-compat: callers that did `embed(x)` still get the manifold
        # position embedding.
        return self.position(x)
