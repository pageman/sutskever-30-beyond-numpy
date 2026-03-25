from __future__ import annotations

from s30bn.paper27_multi_token_prediction import MultiTokenConfig, forward_jax, init_params, params_to_jax, synthetic_mtp_batch


def main() -> None:
    config = MultiTokenConfig()
    contexts, targets = synthetic_mtp_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), contexts, targets)
    print("paper 27 jax loss:", round(float(loss), 6))
    print("paper 27 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
