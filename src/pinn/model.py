from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


class PINNResidualNet(nn.Module):
    """Fully connected residual PINN.

    [HYPERPARAM FIX] Depth and activation are now configurable so the code
    matches the paper (H=4 hidden layers, tanh activation) and so the
    ablation study can sweep them. v1 hard-coded 3 hidden layers with SiLU,
    contradicting the paper text.
    """

    def __init__(self, input_dim: int, hidden: int = 128, output_dim: int = 6,
                 depth: int = 4, activation: str = "tanh") -> None:
        super().__init__()
        act = ACTIVATIONS[activation.lower()]
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, output_dim)
        self.logvar = nn.Linear(hidden, output_dim)
        self.depth = depth
        self.activation = activation

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.head(h), self.logvar(h).clamp(-6.0, 3.0)
