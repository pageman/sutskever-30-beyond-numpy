from __future__ import annotations

from s30bn.paper24_machine_super_intelligence import (
    SuperIntelligenceConfig,
    forward_torch,
    init_params,
    params_to_torch,
    synthetic_superintelligence_batch,
)


def main() -> None:
    config = SuperIntelligenceConfig()
    capabilities, targets = synthetic_superintelligence_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), capabilities, targets)
    print("paper 24 torch loss:", round(float(loss.detach()), 6))
    print("paper 24 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
