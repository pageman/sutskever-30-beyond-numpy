"""Minimal NumPy sanity checks for Attention Is All You Need."""

from __future__ import annotations

from s30bn.paper13_transformer import TransformerConfig, forward_numpy, init_params, synthetic_attention_batch


def main() -> None:
    config = TransformerConfig()
    params = init_params(config)
    seqs, targets = synthetic_attention_batch()
    result = forward_numpy(params, seqs, targets)
    print("paper 13 numpy loss:", round(float(result["loss"]), 6))
    print("paper 13 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
