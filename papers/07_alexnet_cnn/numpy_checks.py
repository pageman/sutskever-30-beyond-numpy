"""Minimal NumPy sanity checks for paper 07."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper07_alexnet import AlexNetConfig, init_params, synthetic_imagenet_like


def main() -> None:
    config = AlexNetConfig()
    images, labels = synthetic_imagenet_like(config)
    params = init_params(config)
    print("images_shape=", images.shape)
    print("labels=", labels.tolist())
    print("conv1_w_shape=", params["conv1_w"].shape)
    print("conv2_w_shape=", params["conv2_w"].shape)


if __name__ == "__main__":
    main()
