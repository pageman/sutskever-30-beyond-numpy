from __future__ import annotations

from s30bn.paper06_pointer_networks import (
    PointerConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_pointer_batch,
)


def main() -> None:
    config = PointerConfig()
    params = params_to_tinygrad(init_params(config))
    seqs, targets = synthetic_pointer_batch()
    loss, logits = forward_tinygrad(params, seqs, targets)
    print("paper 06 tinygrad loss:", round(loss.item(), 6))
    print("paper 06 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
