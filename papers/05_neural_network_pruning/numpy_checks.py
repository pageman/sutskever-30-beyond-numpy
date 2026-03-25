"""Minimal NumPy sanity checks for Keeping Neural Networks Simple."""

from __future__ import annotations

from s30bn.paper05_pruning import PruningConfig, forward_numpy, init_params, synthetic_pruning_batch


def main() -> None:
    config = PruningConfig()
    inputs, targets = synthetic_pruning_batch()
    result = forward_numpy(init_params(config), inputs, targets)
    print("paper 05 numpy loss:", round(float(result["loss"]), 6))
    print("paper 05 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
