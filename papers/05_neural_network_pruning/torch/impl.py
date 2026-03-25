from __future__ import annotations

from s30bn.paper05_pruning import PruningConfig, forward_torch, init_params, params_to_torch, synthetic_pruning_batch


def main() -> None:
    config = PruningConfig()
    inputs, targets = synthetic_pruning_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), inputs, targets)
    print("paper 05 torch loss:", round(float(loss.detach()), 6))
    print("paper 05 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
