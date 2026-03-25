from __future__ import annotations

from s30bn.paper30_lost_in_middle import (
    LostMiddleConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_middle_batch,
)


def main() -> None:
    config = LostMiddleConfig()
    chunks, queries, targets = synthetic_middle_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), chunks, queries, targets)
    print("paper 30 tinygrad loss:", round(loss.item(), 6))
    print("paper 30 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
