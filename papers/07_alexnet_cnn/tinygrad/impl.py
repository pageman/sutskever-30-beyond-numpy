"""tinygrad implementation wrapper for paper 07."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper07_alexnet import AlexNetConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_imagenet_like


def main() -> None:
    config = AlexNetConfig()
    images, labels = synthetic_imagenet_like(config)
    params = params_to_tinygrad(init_params(config))
    loss, logits = forward_tinygrad(params, images, labels)
    print(f"tinygrad loss: {loss.item():.6f}")
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()
