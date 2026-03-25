"""Minimal NumPy sanity checks for Dense Passage Retrieval."""

from __future__ import annotations

from s30bn.paper28_dense_passage_retrieval import DPRConfig, forward_numpy, init_params, synthetic_dpr_batch


def main() -> None:
    config = DPRConfig()
    queries, passages, targets = synthetic_dpr_batch()
    result = forward_numpy(init_params(config), queries, passages, targets)
    print("paper 28 numpy loss:", round(float(result["loss"]), 6))
    print("paper 28 score matrix shape:", result["scores"].shape)


if __name__ == "__main__":
    main()
