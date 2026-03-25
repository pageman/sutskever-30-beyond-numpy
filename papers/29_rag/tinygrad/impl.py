from __future__ import annotations

from s30bn.paper29_rag import RAGConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_rag_batch


def main() -> None:
    config = RAGConfig()
    queries, passages, targets = synthetic_rag_batch()
    loss, probs = forward_tinygrad(params_to_tinygrad(init_params(config)), queries, passages, targets)
    print("paper 29 tinygrad loss:", round(loss.item(), 6))
    print("paper 29 tinygrad probs shape:", tuple(probs.shape))


if __name__ == "__main__":
    main()
