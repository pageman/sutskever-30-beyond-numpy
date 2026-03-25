from __future__ import annotations

from s30bn.paper27_multi_token_prediction import (
    MultiTokenConfig,
    forward_torch,
    init_params,
    params_to_torch,
    synthetic_mtp_batch,
)


def main() -> None:
    config = MultiTokenConfig()
    contexts, targets = synthetic_mtp_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), contexts, targets)
    print("paper 27 torch loss:", round(float(loss.detach()), 6))
    print("paper 27 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
