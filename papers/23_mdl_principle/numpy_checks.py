"""Minimal NumPy sanity checks for MDL Principle."""

from __future__ import annotations

from s30bn.paper23_mdl import MDLConfig, forward_numpy, init_params, synthetic_mdl_batch


def main() -> None:
    config = MDLConfig()
    inputs, targets = synthetic_mdl_batch()
    result = forward_numpy(init_params(config), inputs, targets, config)
    print("paper 23 numpy loss:", round(float(result["loss"]), 6))
    print("paper 23 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
