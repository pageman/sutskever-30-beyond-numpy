from __future__ import annotations

from s30bn.paper25_kolmogorov import KolmogorovConfig, forward_jax, init_params, params_to_jax, synthetic_kolmogorov_batch


def main() -> None:
    config = KolmogorovConfig()
    sequences, targets = synthetic_kolmogorov_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), sequences, targets)
    print("paper 25 jax loss:", round(float(loss), 6))
    print("paper 25 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
