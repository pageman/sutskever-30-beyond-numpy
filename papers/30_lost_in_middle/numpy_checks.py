"""Minimal NumPy sanity checks for Lost in the Middle."""

from __future__ import annotations

from s30bn.paper30_lost_in_middle import LostMiddleConfig, forward_numpy, init_params, synthetic_middle_batch


def main() -> None:
    config = LostMiddleConfig()
    chunks, queries, targets = synthetic_middle_batch()
    result = forward_numpy(init_params(config), chunks, queries, targets)
    print("paper 30 numpy loss:", round(float(result["loss"]), 6))
    print("paper 30 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
