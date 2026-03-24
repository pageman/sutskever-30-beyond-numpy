"""Minimal NumPy sanity checks for paper 03."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper03_lstm import LSTMConfig, build_dataset, forward_numpy, init_params


def main() -> None:
    config = LSTMConfig()
    params = init_params(config)
    inputs, targets = build_dataset(config)
    result = forward_numpy(params, inputs, targets)
    print(f"loss={result['loss']:.6f}")
    print("forget_gate_shape=", result["forget_gate"].shape)


if __name__ == "__main__":
    main()
