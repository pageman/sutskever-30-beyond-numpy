from __future__ import annotations

from s30bn.paper01_complexodynamics import (
    ComplexodynamicsConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_complexodynamics_batch,
)


def main() -> None:
    config = ComplexodynamicsConfig()
    states, targets = synthetic_complexodynamics_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), states, targets)
    print("paper 01 tinygrad loss:", round(loss.item(), 6))
    print("paper 01 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
