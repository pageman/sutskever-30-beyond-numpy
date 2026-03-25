from __future__ import annotations

from s30bn.paper01_complexodynamics import (
    ComplexodynamicsConfig,
    forward_torch,
    init_params,
    params_to_torch,
    synthetic_complexodynamics_batch,
)


def main() -> None:
    config = ComplexodynamicsConfig()
    states, targets = synthetic_complexodynamics_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), states, targets)
    print("paper 01 torch loss:", round(float(loss.detach()), 6))
    print("paper 01 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
