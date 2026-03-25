"""Minimal NumPy sanity checks for paper 14."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper14_bahdanau_attention import BahdanauConfig, forward_numpy, init_params, sample_pair


def main() -> None:
    config = BahdanauConfig()
    source, decoder_token, target = sample_pair(config)
    result = forward_numpy(init_params(config), source, decoder_token, target, config)
    print("source=", source.tolist())
    print("attention=", result["attention"].tolist())
    print(f"loss={result['loss']:.6f}")


if __name__ == "__main__":
    main()
