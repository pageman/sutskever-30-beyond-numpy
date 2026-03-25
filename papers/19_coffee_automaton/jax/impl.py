from __future__ import annotations

from s30bn.paper19_coffee_automaton import CoffeeConfig, forward_jax, init_params, params_to_jax, synthetic_coffee_batch


def main() -> None:
    config = CoffeeConfig()
    states, targets = synthetic_coffee_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), states, targets)
    print("paper 19 jax loss:", round(float(loss), 6))
    print("paper 19 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
