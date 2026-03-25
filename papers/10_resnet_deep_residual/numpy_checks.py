"""Minimal NumPy sanity checks for paper 10."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper10_resnet import ResNetConfig, init_params, synthetic_residual_dataset


def main() -> None:
    config = ResNetConfig()
    images, labels = synthetic_residual_dataset(config)
    params = init_params(config)
    print("images_shape=", images.shape)
    print("labels=", labels.tolist())
    print("stem_w_shape=", params["stem_w"].shape)
    print("block1_w_shape=", params["block1_w"].shape)


if __name__ == "__main__":
    main()
