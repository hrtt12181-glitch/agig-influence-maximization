"""Dynamic social graph encoder using weighted GraphSAGE plus temporal GRU."""

from __future__ import annotations

import torch
from torch import nn


class GraphSAGELayer(nn.Module):
    """A small weighted mean-aggregation GraphSAGE layer.

    Inputs:
        node_features: [U, D]
        edge_index: [2, E], directed edges source -> target
        edge_weight: [E]

    Output:
        updated node features: [U, D]
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.self_proj = nn.Linear(model_dim, model_dim)
        self.neighbor_proj = nn.Linear(model_dim, model_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        num_nodes = node_features.size(0)
        src, dst = edge_index
        weighted_messages = node_features[src] * edge_weight.view(-1, 1)
        neighbor_sum = torch.zeros_like(node_features)
        degree = torch.zeros(num_nodes, 1, device=node_features.device)
        neighbor_sum.index_add_(0, dst, weighted_messages)
        degree.index_add_(0, dst, edge_weight.view(-1, 1))
        neighbor_mean = neighbor_sum / degree.clamp_min(1e-6)
        return torch.relu(self.self_proj(node_features) + self.neighbor_proj(neighbor_mean))


class DynamicSocialGraphEncoder(nn.Module):
    """Encode a sequence of social graph snapshots.

    Inputs:
        initial_node_features: [U, D]
        edge_index_seq: list of T tensors, each [2, E_t]
        edge_weight_seq: list of T tensors, each [E_t]

    Output:
        graph_context: [U, D] dynamic user representation g_u^t
    """

    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.sage = GraphSAGELayer(model_dim)
        self.temporal_gru = nn.GRU(model_dim, model_dim, batch_first=True)

    def forward(
        self,
        initial_node_features: torch.Tensor,
        edge_index_seq: list[torch.Tensor],
        edge_weight_seq: list[torch.Tensor],
    ) -> torch.Tensor:
        snapshot_embeddings = []
        h = initial_node_features
        for edge_index, edge_weight in zip(edge_index_seq, edge_weight_seq):
            h = self.sage(h, edge_index.to(h.device), edge_weight.to(h.device))
            snapshot_embeddings.append(h)
        sequence = torch.stack(snapshot_embeddings, dim=1)
        output, _ = self.temporal_gru(sequence)
        return output[:, -1, :]
