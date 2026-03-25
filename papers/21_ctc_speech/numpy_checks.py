"""Minimal NumPy sanity checks for paper 21."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper21_ctc import CTCConfig, forward_numpy, init_params, synthetic_ctc_batch


def main() -> None:
    config = CTCConfig()
    features, targets = synthetic_ctc_batch()
    result = forward_numpy(init_params(config), features, targets, config)
    print("features_shape=", features.shape)
    print("targets=", targets.tolist())
    print(f"loss={result['loss']:.6f}")


if __name__ == "__main__":
    main()
