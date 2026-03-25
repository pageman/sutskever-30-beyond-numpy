"""PyTorch implementation wrapper for paper 11."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper11_dilated_convolutions import (
    DilatedConvConfig,
    forward_torch,
    init_params,
    params_to_torch,
    synthetic_context_dataset,
)


def main() -> None:
    config = DilatedConvConfig()
    images, labels = synthetic_context_dataset(config)
    params = params_to_torch(init_params(config))
    loss, logits = forward_torch(params, images, labels, config)
    print(f"torch loss: {loss.item():.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
