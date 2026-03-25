from __future__ import annotations

from s30bn.paper24_machine_super_intelligence import (
    SuperIntelligenceConfig,
    forward_jax,
    init_params,
    params_to_jax,
    synthetic_superintelligence_batch,
)


def main() -> None:
    config = SuperIntelligenceConfig()
    capabilities, targets = synthetic_superintelligence_batch()
    loss, logits = forward_jax(params_to_jax(init_params(config)), capabilities, targets)
    print("paper 24 jax loss:", round(float(loss), 6))
    print("paper 24 jax logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
