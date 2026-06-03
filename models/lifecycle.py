"""Interest lifecycle estimation and lifecycle-aware interest transformation."""

from __future__ import annotations

import torch
from torch import nn


class InterestLifecycleModule(nn.Module):
    """Predict rising/stable/declining lifecycle states for interests.

    Inputs:
        interests: [U, K, D]
        lifecycle_features: [U, 5] ordered as
            recent_activity, frequency_trend, recency, interaction_interval, growth_slope

    Outputs:
        pi: [U, 3] probabilities for rising, stable, declining
        h_tilde: [U, K, D] lifecycle-transformed interest slots
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.state_classifier = nn.Sequential(nn.Linear(5, model_dim), nn.ReLU(), nn.Linear(model_dim, 3))
        self.rising = nn.Linear(model_dim, model_dim)
        self.stable = nn.Linear(model_dim, model_dim)
        self.declining = nn.Linear(model_dim, model_dim)

    def forward(self, interests: torch.Tensor, lifecycle_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pi = torch.softmax(self.state_classifier(lifecycle_features), dim=-1)
        rising_h = self.rising(interests)
        stable_h = self.stable(interests)
        declining_h = self.declining(interests)
        h_tilde = (
            pi[:, 0].view(-1, 1, 1) * rising_h
            + pi[:, 1].view(-1, 1, 1) * stable_h
            + pi[:, 2].view(-1, 1, 1) * declining_h
        )
        return pi, h_tilde
