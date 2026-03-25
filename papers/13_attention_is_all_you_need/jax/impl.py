from __future__ import annotations

from s30bn.paper13_transformer import TransformerConfig, forward_jax, init_params, params_to_jax, synthetic_attention_batch


def main() -> None:
    config = TransformerConfig()
    params = params_to_jax(init_params(config))
    seqs, targets = synthetic_attention_batch()
    loss, logits = forward_jax(params, seqs, targets)
    print("paper 13 jax loss:", round(float(loss), 6))
    print("paper 13 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
