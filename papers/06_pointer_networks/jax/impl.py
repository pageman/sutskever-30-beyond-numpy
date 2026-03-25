from __future__ import annotations

from s30bn.paper06_pointer_networks import PointerConfig, forward_jax, init_params, params_to_jax, synthetic_pointer_batch


def main() -> None:
    config = PointerConfig()
    params = params_to_jax(init_params(config))
    seqs, targets = synthetic_pointer_batch()
    loss, logits = forward_jax(params, seqs, targets)
    print("paper 06 jax loss:", round(float(loss), 6))
    print("paper 06 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
