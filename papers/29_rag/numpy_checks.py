"""Minimal NumPy sanity checks for Retrieval-Augmented Generation."""

from __future__ import annotations

from s30bn.paper29_rag import RAGConfig, forward_numpy, init_params, synthetic_rag_batch


def main() -> None:
    config = RAGConfig()
    queries, passages, targets = synthetic_rag_batch()
    result = forward_numpy(init_params(config), queries, passages, targets)
    print("paper 29 numpy loss:", round(float(result["loss"]), 6))
    print("paper 29 probs shape:", result["probs"].shape)


if __name__ == "__main__":
    main()
