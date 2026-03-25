from __future__ import annotations

from s30bn.paper06_pointer_networks import (
    PointerConfig,
    forward_torch,
    init_params,
    params_to_torch,
    synthetic_pointer_batch,
)


def main() -> None:
    config = PointerConfig()
    params = params_to_torch(init_params(config))
    seqs, targets = synthetic_pointer_batch()
    loss, logits = forward_torch(params, seqs, targets)
    print("paper 06 torch loss:", round(float(loss.detach()), 6))
    print("paper 06 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
