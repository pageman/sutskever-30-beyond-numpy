from __future__ import annotations

from s30bn.paper13_transformer import (
    TransformerConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_attention_batch,
)


def main() -> None:
    config = TransformerConfig()
    params = params_to_tinygrad(init_params(config))
    seqs, targets = synthetic_attention_batch()
    loss, logits = forward_tinygrad(params, seqs, targets)
    print("paper 13 tinygrad loss:", round(loss.item(), 6))
    print("paper 13 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
