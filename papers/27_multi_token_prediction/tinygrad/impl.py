from __future__ import annotations

from s30bn.paper27_multi_token_prediction import (
    MultiTokenConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_mtp_batch,
)


def main() -> None:
    config = MultiTokenConfig()
    contexts, targets = synthetic_mtp_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), contexts, targets)
    print("paper 27 tinygrad loss:", round(loss.item(), 6))
    print("paper 27 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
