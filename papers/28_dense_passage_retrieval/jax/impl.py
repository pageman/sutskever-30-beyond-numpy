from __future__ import annotations

from s30bn.paper28_dense_passage_retrieval import DPRConfig, forward_jax, init_params, params_to_jax, synthetic_dpr_batch


def main() -> None:
    config = DPRConfig()
    queries, passages, targets = synthetic_dpr_batch()
    loss, scores = forward_jax(params_to_jax(init_params(config)), queries, passages, targets)
    print("paper 28 jax loss:", round(float(loss), 6))
    print("paper 28 jax score shape:", tuple(scores.shape))


if __name__ == "__main__":
    main()
