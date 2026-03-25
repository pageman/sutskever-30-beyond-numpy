"""Minimal NumPy sanity checks for The Coffee Automaton."""

from __future__ import annotations

from s30bn.paper19_coffee_automaton import CoffeeConfig, forward_numpy, init_params, synthetic_coffee_batch


def main() -> None:
    config = CoffeeConfig()
    states, targets = synthetic_coffee_batch()
    result = forward_numpy(init_params(config), states, targets)
    print("paper 19 numpy loss:", round(float(result["loss"]), 6))
    print("paper 19 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
