"""Target-aware propagation probability module."""

from __future__ import annotations

import torch
from torch import nn


class TargetAwarePropagation(nn.Module):
    """Estimate edge-level target-aware propagation probabilities.

    Inputs:
        graph_context: [U, D]
        user_content_repr: [U, D]
        edge_index: [2, E]
        edge_weight: [E]
        interest_score: [U]

    Outputs:
        p_diff: [E] generic diffusion probability
        p_target: [E] p_diff * I_v(c,t), for edge u -> v
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(model_dim * 4 + 1, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        graph_context: torch.Tensor,
        user_content_repr: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        interest_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_features = torch.cat(
            [
                graph_context[src],
                graph_context[dst],
                user_content_repr[src],
                user_content_repr[dst],
                edge_weight.view(-1, 1),
            ],
            dim=-1,
        )
        p_diff = torch.sigmoid(self.edge_mlp(edge_features)).squeeze(-1)
        p_target = p_diff * interest_score[dst]
        return p_diff, p_target
