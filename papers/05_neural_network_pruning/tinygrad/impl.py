from __future__ import annotations

from s30bn.paper05_pruning import PruningConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_pruning_batch


def main() -> None:
    config = PruningConfig()
    inputs, targets = synthetic_pruning_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), inputs, targets)
    print("paper 05 tinygrad loss:", round(loss.item(), 6))
    print("paper 05 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
