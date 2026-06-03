"""Toy data generator for the dynamic social influence maximization prototype.

All tensors are intentionally small and random so the full pipeline can run on CPU.
"""

from __future__ import annotations

import torch


def build_toy_data(seed: int = 7) -> dict:
    """Build a deterministic toy dataset.

    Returns:
        A dictionary with these shapes:
        - user_history_text: [num_users, history_len, text_dim]
        - target_content_text: [text_dim]
        - user_history_entities: [num_users, history_len, entities_per_content]
        - target_content_entities: [entities_per_content]
        - kg_triples: [num_triples, 3] storing (head, relation, tail)
        - user_static_features: [num_users, model_dim]
        - lifecycle_features: [num_users, 5]
        - kg_similarity: [num_users, 1]
        - activity: [num_users, 1]
        - edge_index_seq: list of [2, num_edges_t]
        - edge_weight_seq: list of [num_edges_t]
    """

    generator = torch.Generator().manual_seed(seed)
    num_users = 8
    history_len = 5
    text_dim = 16
    model_dim = 32
    num_entities = 14
    num_relations = 4
    entities_per_content = 3

    user_history_text = torch.randn(num_users, history_len, text_dim, generator=generator)
    target_content_text = torch.randn(text_dim, generator=generator)

    user_history_entities = torch.randint(
        0, num_entities, (num_users, history_len, entities_per_content), generator=generator
    )
    target_content_entities = torch.tensor([1, 4, 9], dtype=torch.long)

    kg_triples = torch.tensor(
        [
            [0, 0, 1],
            [1, 1, 2],
            [2, 0, 3],
            [3, 2, 4],
            [4, 1, 5],
            [5, 3, 6],
            [6, 2, 7],
            [7, 0, 8],
            [8, 1, 9],
            [9, 3, 10],
            [10, 2, 11],
            [11, 0, 12],
            [12, 1, 13],
            [13, 3, 0],
            [1, 2, 9],
            [4, 0, 10],
        ],
        dtype=torch.long,
    )

    user_static_features = torch.randn(num_users, model_dim, generator=generator)

    # [recent_activity, frequency_trend, recency, interaction_interval, growth_slope]
    lifecycle_features = torch.rand(num_users, 5, generator=generator)
    kg_similarity = torch.rand(num_users, 1, generator=generator)
    activity = torch.rand(num_users, 1, generator=generator)

    edge_index_seq = [
        torch.tensor(
            [[0, 0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 3, 4, 5, 6, 7]], dtype=torch.long
        ),
        torch.tensor(
            [[0, 1, 1, 2, 2, 3, 4, 5, 6], [2, 2, 4, 3, 5, 5, 6, 6, 7]], dtype=torch.long
        ),
        torch.tensor(
            [[0, 1, 2, 2, 3, 4, 4, 5, 6, 7], [1, 3, 3, 4, 5, 5, 6, 7, 7, 0]],
            dtype=torch.long,
        ),
    ]
    edge_weight_seq = [torch.rand(edges.size(1), generator=generator) * 0.7 + 0.3 for edges in edge_index_seq]

    return {
        "num_users": num_users,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "history_len": history_len,
        "text_dim": text_dim,
        "model_dim": model_dim,
        "user_history_text": user_history_text,
        "target_content_text": target_content_text,
        "user_history_entities": user_history_entities,
        "target_content_entities": target_content_entities,
        "kg_triples": kg_triples,
        "user_static_features": user_static_features,
        "lifecycle_features": lifecycle_features,
        "kg_similarity": kg_similarity,
        "activity": activity,
        "edge_index_seq": edge_index_seq,
        "edge_weight_seq": edge_weight_seq,
    }
