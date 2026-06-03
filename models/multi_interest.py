"""Attention-based multi-interest extraction."""

from __future__ import annotations

import torch
from torch import nn


class MultiInterestExtractor(nn.Module):
    """Extract K latent interest slots from a user's content history.

    Input:
        content_sequence: [U, L, D]

    Output:
        interests: [U, K, D]
    """

    def __init__(self, model_dim: int, num_interests: int) -> None:
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(num_interests, model_dim) * 0.02)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.scale = model_dim**-0.5

    def forward(self, content_sequence: torch.Tensor) -> torch.Tensor:
        keys = self.key(content_sequence)
        values = self.value(content_sequence)
        logits = torch.einsum("kd,uld->ukl", self.slot_queries, keys) * self.scale
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("ukl,uld->ukd", weights, values)
