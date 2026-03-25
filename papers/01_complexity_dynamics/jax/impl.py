from __future__ import annotations

from s30bn.paper01_complexodynamics import ComplexodynamicsConfig, forward_jax, init_params, params_to_jax, synthetic_complexodynamics_batch


def main() -> None:
    config = ComplexodynamicsConfig()
    states, targets = synthetic_complexodynamics_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), states, targets)
    print("paper 01 jax loss:", round(float(loss), 6))
    print("paper 01 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
