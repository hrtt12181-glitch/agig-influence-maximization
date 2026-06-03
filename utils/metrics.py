"""Small metrics/helpers for reporting prototype outputs."""

from __future__ import annotations

import torch


def edge_probabilities_to_matrix(num_users: int, edge_index: torch.Tensor, edge_probability: torch.Tensor) -> torch.Tensor:
    """Convert sparse edge probabilities to a dense [U, U] matrix."""
    matrix = torch.zeros(num_users, num_users, device=edge_probability.device)
    src, dst = edge_index
    matrix[src, dst] = edge_probability
    return matrix


def top_interest_users(interest_score: torch.Tensor, top_n: int = 3) -> list[tuple[int, float]]:
    """Return the top-N users by I_u(c,t)."""
    values, indices = torch.topk(interest_score, k=min(top_n, interest_score.numel()))
    return [(int(index), float(value)) for index, value in zip(indices, values)]
