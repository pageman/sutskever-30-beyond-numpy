from __future__ import annotations

from s30bn.paper29_rag import RAGConfig, forward_torch, init_params, params_to_torch, synthetic_rag_batch


def main() -> None:
    config = RAGConfig()
    queries, passages, targets = synthetic_rag_batch()
    loss, probs = forward_torch(params_to_torch(init_params(config)), queries, passages, targets)
    print("paper 29 torch loss:", round(float(loss.detach()), 6))
    print("paper 29 torch probs shape:", tuple(probs.shape))


if __name__ == "__main__":
    main()
