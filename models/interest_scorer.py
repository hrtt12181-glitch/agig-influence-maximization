"""Potential interest user scorer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PotentialInterestUserScorer(nn.Module):
    """Score user interest in target content and build content-aware user vectors.

    Inputs:
        q_c: [D]
        interests: [U, K, D]
        lifecycle_pi: [U, 3]
        kg_similarity: [U, 1]
        activity: [U, 1]
        graph_context: [U, D]

    Outputs:
        interest_score: [U] in [0, 1], i.e. I_u(c,t)
        user_content_repr: [U, D], i.e. z_u^{c,t}
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        scalar_dim = 1 + 3 + 1 + 1
        self.scorer = nn.Sequential(
            nn.Linear(model_dim * 2 + scalar_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, 1),
        )
        self.z_fusion = nn.Sequential(nn.Linear(model_dim * 3, model_dim), nn.ReLU(), nn.Linear(model_dim, model_dim))

    def forward(
        self,
        q_c: torch.Tensor,
        interests: torch.Tensor,
        lifecycle_pi: torch.Tensor,
        kg_similarity: torch.Tensor,
        activity: torch.Tensor,
        graph_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_users = interests.size(0)
        q = q_c.view(1, 1, -1).expand(num_users, interests.size(1), -1)
        slot_similarity = F.cosine_similarity(interests, q, dim=-1)
        max_similarity, best_slot = slot_similarity.max(dim=-1)
        best_interest = interests[torch.arange(num_users, device=interests.device), best_slot]

        q_flat = q_c.view(1, -1).expand(num_users, -1)
        features = torch.cat(
            [q_flat, best_interest, max_similarity.unsqueeze(-1), lifecycle_pi, kg_similarity, activity], dim=-1
        )
        interest_score = torch.sigmoid(self.scorer(features)).squeeze(-1)
        user_content_repr = self.z_fusion(torch.cat([q_flat, best_interest, graph_context], dim=-1))
        return interest_score, user_content_repr
