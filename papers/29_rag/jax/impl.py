from __future__ import annotations

from s30bn.paper29_rag import RAGConfig, forward_jax, init_params, params_to_jax, synthetic_rag_batch


def main() -> None:
    config = RAGConfig()
    queries, passages, targets = synthetic_rag_batch()
    loss, probs = forward_jax(params_to_jax(init_params(config)), queries, passages, targets)
    print("paper 29 jax loss:", round(float(loss), 6))
    print("paper 29 jax probs shape:", tuple(probs.shape))


if __name__ == "__main__":
    main()
