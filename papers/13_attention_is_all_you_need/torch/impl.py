from __future__ import annotations

from s30bn.paper13_transformer import (
    TransformerConfig,
    forward_torch,
    init_params,
    params_to_torch,
    synthetic_attention_batch,
)


def main() -> None:
    config = TransformerConfig()
    params = params_to_torch(init_params(config))
    seqs, targets = synthetic_attention_batch()
    loss, logits = forward_torch(params, seqs, targets)
    print("paper 13 torch loss:", round(float(loss.detach()), 6))
    print("paper 13 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
