from __future__ import annotations

from s30bn.paper23_mdl import MDLConfig, forward_torch, init_params, params_to_torch, synthetic_mdl_batch


def main() -> None:
    config = MDLConfig()
    inputs, targets = synthetic_mdl_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), inputs, targets, config)
    print("paper 23 torch loss:", round(float(loss.detach()), 6))
    print("paper 23 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
