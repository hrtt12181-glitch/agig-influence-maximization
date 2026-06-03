

from __future__ import annotations

import torch

from data.toy_data import build_toy_data
from models.dynamic_graph_encoder import DynamicSocialGraphEncoder
from models.interest_scorer import PotentialInterestUserScorer
from models.kg_encoder import KGEnhancedContentEncoder
from models.lifecycle import InterestLifecycleModule
from models.multi_interest import MultiInterestExtractor
from models.propagation import TargetAwarePropagation
from optimization.seed_selection import greedy_seed_selection
from utils.metrics import edge_probabilities_to_matrix, top_interest_users


def main() -> None:
    torch.manual_seed(42)
    data = build_toy_data(seed=7)
    k = 3
    num_interests = 4

    kg_encoder = KGEnhancedContentEncoder(data["num_entities"], data["text_dim"], data["model_dim"])
    multi_interest = MultiInterestExtractor(data["model_dim"], num_interests)
    lifecycle = InterestLifecycleModule(data["model_dim"])
    graph_encoder = DynamicSocialGraphEncoder(data["model_dim"])
    scorer = PotentialInterestUserScorer(data["model_dim"])
    propagation = TargetAwarePropagation(data["model_dim"])

    q_c, history_repr = kg_encoder(
        data["user_history_text"],
        data["target_content_text"],
        data["user_history_entities"],
        data["target_content_entities"],
        data["kg_triples"],
    )
    interests = multi_interest(history_repr)
    lifecycle_pi, lifecycle_interests = lifecycle(interests, data["lifecycle_features"])
    graph_context = graph_encoder(data["user_static_features"], data["edge_index_seq"], data["edge_weight_seq"])
    interest_score, user_content_repr = scorer(
        q_c,
        lifecycle_interests,
        lifecycle_pi,
        data["kg_similarity"],
        data["activity"],
        graph_context,
    )

    current_edge_index = data["edge_index_seq"][-1]
    current_edge_weight = data["edge_weight_seq"][-1]
    p_diff, p_target = propagation(
        graph_context, user_content_repr, current_edge_index, current_edge_weight, interest_score
    )
    propagation_matrix = edge_probabilities_to_matrix(data["num_users"], current_edge_index, p_target)
    seeds, objective_value = greedy_seed_selection(propagation_matrix, interest_score, k=k, hops=2)

    print("=== Potential interest score I_v(c,t) ===")
    for user_id, score in enumerate(interest_score.detach().tolist()):
        print(f"user {user_id}: {score:.4f}")

    print("\n=== Target-aware propagation probabilities p_target(u,v,c,t) ===")
    for edge_id, (src, dst) in enumerate(current_edge_index.t().tolist()):
        print(
            f"edge {src}->{dst}: p_diff={p_diff[edge_id].item():.4f}, "
            f"p_target={p_target[edge_id].item():.4f}"
        )

    print("\n=== Greedy top-k seed users ===")
    print(f"top interest users: {top_interest_users(interest_score, top_n=k)}")
    print(f"selected seeds (k={k}): {seeds}")
    print(f"approx F(S): {objective_value:.4f}")


if __name__ == "__main__":
    main()
