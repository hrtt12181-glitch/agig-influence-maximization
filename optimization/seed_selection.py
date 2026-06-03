"""Greedy seed selection for targeted influence maximization."""

from __future__ import annotations

import torch


def approximate_target_influence(
    seeds: list[int], propagation_matrix: torch.Tensor, interest_score: torch.Tensor, hops: int = 2
) -> float:
    """Approximate F(S) with one-hop or two-hop independent activation paths.

    Args:
        seeds: selected seed node ids.
        propagation_matrix: [U, U], p_target(u, v, c, t) for directed edges; zero for missing edges.
        interest_score: [U], I_v(c,t).
        hops: 1 or 2 propagation hops.

    Returns:
        Scalar approximate targeted influence score.
    """
    if not seeds:
        return 0.0

    num_users = propagation_matrix.size(0)
    seed_tensor = torch.tensor(seeds, dtype=torch.long, device=propagation_matrix.device)
    activated_prob = torch.zeros(num_users, device=propagation_matrix.device)
    activated_prob[seed_tensor] = 1.0

    one_hop_fail = torch.prod(1.0 - propagation_matrix[seed_tensor], dim=0)
    one_hop_prob = 1.0 - one_hop_fail
    activated_prob = torch.maximum(activated_prob, one_hop_prob)

    if hops >= 2:
        two_hop_paths = one_hop_prob.unsqueeze(1) * propagation_matrix
        two_hop_prob = 1.0 - torch.prod(1.0 - two_hop_paths.clamp(0.0, 1.0), dim=0)
        activated_prob = torch.maximum(activated_prob, two_hop_prob)

    return float((activated_prob * interest_score).sum().item())


def greedy_seed_selection(
    propagation_matrix: torch.Tensor, interest_score: torch.Tensor, k: int, hops: int = 2
) -> tuple[list[int], float]:
    """Select k seeds by greedy marginal gain.

    Inputs:
        propagation_matrix: [U, U]
        interest_score: [U]
        k: number of seeds
        hops: 1 or 2 for the approximation depth

    Outputs:
        seeds: list[int]
        best_score: approximate F(S)
    """
    selected: list[int] = []
    candidates = set(range(propagation_matrix.size(0)))
    current_score = 0.0

    for _ in range(k):
        best_node = None
        best_score = -1.0
        for node in candidates:
            score = approximate_target_influence(selected + [node], propagation_matrix, interest_score, hops=hops)
            if score > best_score:
                best_score = score
                best_node = node
        if best_node is None:
            break
        selected.append(best_node)
        candidates.remove(best_node)
        current_score = best_score

    return selected, current_score
