from __future__ import annotations

from s30bn.paper28_dense_passage_retrieval import DPRConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_dpr_batch


def main() -> None:
    config = DPRConfig()
    queries, passages, targets = synthetic_dpr_batch()
    loss, scores = forward_tinygrad(params_to_tinygrad(init_params(config)), queries, passages, targets)
    print("paper 28 tinygrad loss:", round(loss.item(), 6))
    print("paper 28 tinygrad score shape:", tuple(scores.shape))


if __name__ == "__main__":
    main()
