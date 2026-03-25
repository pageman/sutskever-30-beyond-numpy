from __future__ import annotations

from s30bn.paper25_kolmogorov import (
    KolmogorovConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_kolmogorov_batch,
)


def main() -> None:
    config = KolmogorovConfig()
    sequences, targets = synthetic_kolmogorov_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), sequences, targets)
    print("paper 25 tinygrad loss:", round(loss.item(), 6))
    print("paper 25 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
