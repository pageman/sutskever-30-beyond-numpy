"""Minimal NumPy sanity checks for paper 20."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper20_neural_turing_machine import NTMConfig, build_dataset, forward_numpy, init_params


def main() -> None:
    config = NTMConfig()
    inputs, targets = build_dataset(config)
    result = forward_numpy(init_params(config), inputs, targets, config)
    print("inputs=", inputs.tolist())
    print("targets=", targets.tolist())
    print("weights_shape=", result["weights"].shape)


if __name__ == "__main__":
    main()
