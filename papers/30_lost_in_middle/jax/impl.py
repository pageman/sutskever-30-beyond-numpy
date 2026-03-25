from __future__ import annotations

from s30bn.paper30_lost_in_middle import LostMiddleConfig, forward_jax, init_params, params_to_jax, synthetic_middle_batch


def main() -> None:
    config = LostMiddleConfig()
    chunks, queries, targets = synthetic_middle_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), chunks, queries, targets)
    print("paper 30 jax loss:", round(float(loss), 6))
    print("paper 30 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
