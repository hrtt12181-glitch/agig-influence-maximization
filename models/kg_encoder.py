"""KG-enhanced content encoder."""

from __future__ import annotations

import torch
from torch import nn


class KGEnhancedContentEncoder(nn.Module):
    """Fuse random text embeddings with KG entity-neighborhood representations.

    Inputs:
        history_text: [U, L, text_dim]
        target_text: [text_dim]
        history_entity_ids: [U, L, M]
        target_entity_ids: [M]
        kg_triples: [T, 3] as (head, relation, tail)

    Outputs:
        q_c: [D] target content representation
        history_repr: [U, L, D] historical content representations
    """

    def __init__(self, num_entities: int, text_dim: int, model_dim: int) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, model_dim)
        self.neighbor_proj = nn.Linear(model_dim, model_dim)
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def _entity_representations(self, kg_triples: torch.Tensor) -> torch.Tensor:
        """Return KG-smoothed entity embeddings with shape [num_entities, D]."""
        base = self.entity_embedding.weight
        num_entities = base.size(0)
        heads = kg_triples[:, 0]
        tails = kg_triples[:, 2]

        neighbor_sum = torch.zeros_like(base)
        degree = torch.zeros(num_entities, 1, device=base.device)
        neighbor_sum.index_add_(0, heads, base[tails])
        neighbor_sum.index_add_(0, tails, base[heads])
        degree.index_add_(0, heads, torch.ones_like(degree[heads]))
        degree.index_add_(0, tails, torch.ones_like(degree[tails]))
        neighbor_mean = neighbor_sum / degree.clamp_min(1.0)
        return base + self.neighbor_proj(neighbor_mean)

    def _encode_content(
        self, text_embeddings: torch.Tensor, entity_ids: torch.Tensor, kg_triples: torch.Tensor
    ) -> torch.Tensor:
        """Encode arbitrary leading dimensions ending in text/entity axes.

        Args:
            text_embeddings: [..., text_dim]
            entity_ids: [..., M]
            kg_triples: [T, 3]

        Returns:
            content representations: [..., D]
        """
        entity_table = self._entity_representations(kg_triples)
        entity_repr = entity_table[entity_ids].mean(dim=-2)
        return self.fusion(torch.cat([text_embeddings, entity_repr], dim=-1))

    def forward(
        self,
        history_text: torch.Tensor,
        target_text: torch.Tensor,
        history_entity_ids: torch.Tensor,
        target_entity_ids: torch.Tensor,
        kg_triples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_repr = self._encode_content(history_text, history_entity_ids, kg_triples)
        q_c = self._encode_content(target_text, target_entity_ids, kg_triples)
        return q_c, history_repr
