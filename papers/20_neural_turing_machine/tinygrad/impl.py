"""tinygrad implementation wrapper for paper 20."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper20_neural_turing_machine import (
    NTMConfig,
    build_dataset,
    forward_tinygrad,
    init_params,
    params_to_tinygrad,
)


def main() -> None:
    config = NTMConfig()
    inputs, targets = build_dataset(config)
    params = params_to_tinygrad(init_params(config))
    loss, logits = forward_tinygrad(params, inputs, targets, config)
    print(f"tinygrad loss: {loss.item():.6f}")
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()
