"""Neural network models for arrival time prediction."""

from __future__ import annotations

import torch
from torch import nn


class ArrivalPredictor(nn.Module):
    """Predict per-flight arrival time adjustments based on features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta = self.net(features)
        return delta.squeeze(-1)


__all__ = ["ArrivalPredictor"]
