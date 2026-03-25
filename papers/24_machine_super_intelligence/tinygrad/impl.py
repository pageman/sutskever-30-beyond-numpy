from __future__ import annotations

from s30bn.paper24_machine_super_intelligence import (
    SuperIntelligenceConfig,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
    synthetic_superintelligence_batch,
)


def main() -> None:
    config = SuperIntelligenceConfig()
    capabilities, targets = synthetic_superintelligence_batch()
    loss, logits = forward_tinygrad(params_to_tinygrad(init_params(config)), capabilities, targets)
    print("paper 24 tinygrad loss:", round(loss.item(), 6))
    print("paper 24 tinygrad logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
