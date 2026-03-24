"""Minimal NumPy sanity checks for paper 26."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper26_cs231n import CNNConfig, init_params, synthetic_cifar_like


def main() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = init_params(config)
    print("images_shape=", images.shape)
    print("labels=", labels.tolist())
    print("conv_w_shape=", params["conv_w"].shape)


if __name__ == "__main__":
    main()
