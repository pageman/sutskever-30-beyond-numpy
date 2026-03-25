"""Minimal NumPy sanity checks for Kolmogorov Complexity."""

from __future__ import annotations

from s30bn.paper25_kolmogorov import KolmogorovConfig, forward_numpy, init_params, synthetic_kolmogorov_batch


def main() -> None:
    config = KolmogorovConfig()
    sequences, targets = synthetic_kolmogorov_batch()
    result = forward_numpy(init_params(config), sequences, targets)
    print("paper 25 numpy loss:", round(float(result["loss"]), 6))
    print("paper 25 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
