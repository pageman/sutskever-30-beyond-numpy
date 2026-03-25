"""Minimal NumPy sanity checks for Multi-token Prediction."""

from __future__ import annotations

from s30bn.paper27_multi_token_prediction import MultiTokenConfig, forward_numpy, init_params, synthetic_mtp_batch


def main() -> None:
    config = MultiTokenConfig()
    contexts, targets = synthetic_mtp_batch()
    result = forward_numpy(init_params(config), contexts, targets)
    print("paper 27 numpy loss:", round(float(result["loss"]), 6))
    print("paper 27 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
