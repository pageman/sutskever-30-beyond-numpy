from __future__ import annotations

from s30bn.paper19_coffee_automaton import CoffeeConfig, forward_torch, init_params, params_to_torch, synthetic_coffee_batch


def main() -> None:
    config = CoffeeConfig()
    states, targets = synthetic_coffee_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), states, targets)
    print("paper 19 torch loss:", round(float(loss.detach()), 6))
    print("paper 19 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
