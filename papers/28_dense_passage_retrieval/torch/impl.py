from __future__ import annotations

from s30bn.paper28_dense_passage_retrieval import DPRConfig, forward_torch, init_params, params_to_torch, synthetic_dpr_batch


def main() -> None:
    config = DPRConfig()
    queries, passages, targets = synthetic_dpr_batch()
    loss, scores = forward_torch(params_to_torch(init_params(config)), queries, passages, targets)
    print("paper 28 torch loss:", round(float(loss.detach()), 6))
    print("paper 28 torch score shape:", tuple(scores.shape))


if __name__ == "__main__":
    main()
