from __future__ import annotations

from s30bn.paper23_mdl import MDLConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_mdl_batch


def main() -> None:
    config = MDLConfig()
    inputs, targets = synthetic_mdl_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), inputs, targets, config)
    print("paper 23 tinygrad loss:", round(loss.item(), 6))
    print("paper 23 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
