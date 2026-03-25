from __future__ import annotations

from s30bn.paper23_mdl import MDLConfig, forward_jax, init_params, params_to_jax, synthetic_mdl_batch


def main() -> None:
    config = MDLConfig()
    inputs, targets = synthetic_mdl_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), inputs, targets, config)
    print("paper 23 jax loss:", round(float(loss), 6))
    print("paper 23 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
