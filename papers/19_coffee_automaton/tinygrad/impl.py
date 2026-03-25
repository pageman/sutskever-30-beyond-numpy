from __future__ import annotations

from s30bn.paper19_coffee_automaton import CoffeeConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_coffee_batch


def main() -> None:
    config = CoffeeConfig()
    states, targets = synthetic_coffee_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), states, targets)
    print("paper 19 tinygrad loss:", round(loss.item(), 6))
    print("paper 19 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
