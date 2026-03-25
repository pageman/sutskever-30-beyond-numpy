from __future__ import annotations

from s30bn.paper05_pruning import PruningConfig, forward_jax, init_params, params_to_jax, synthetic_pruning_batch


def main() -> None:
    config = PruningConfig()
    inputs, targets = synthetic_pruning_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), inputs, targets)
    print("paper 05 jax loss:", round(float(loss), 6))
    print("paper 05 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
