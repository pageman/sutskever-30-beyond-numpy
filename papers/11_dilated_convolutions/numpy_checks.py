"""Minimal NumPy sanity checks for paper 11."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper11_dilated_convolutions import DilatedConvConfig, init_params, synthetic_context_dataset


def main() -> None:
    config = DilatedConvConfig()
    images, labels = synthetic_context_dataset(config)
    params = init_params(config)
    print("images_shape=", images.shape)
    print("labels=", labels.tolist())
    print("dilated_kernel_shape=", params["dilated_w"].shape)


if __name__ == "__main__":
    main()
