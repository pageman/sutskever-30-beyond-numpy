"""PyTorch implementation wrapper for paper 04."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper04_rnn_regularization import (
    RNNRegularizationConfig,
    build_dataset,
    forward_torch,
    init_params,
    params_to_torch,
)


def main() -> None:
    config = RNNRegularizationConfig()
    inputs, targets = build_dataset(config)
    params = params_to_torch(init_params(config))
    loss, logits = forward_torch(params, inputs, targets, config)
    print(f"torch loss: {loss.item():.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
