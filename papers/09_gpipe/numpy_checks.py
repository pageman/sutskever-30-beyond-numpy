"""Minimal NumPy sanity checks for paper 09."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper09_gpipe import GPipeConfig, forward_numpy, init_params, synthetic_pipeline_batch


def main() -> None:
    config = GPipeConfig()
    batch, labels = synthetic_pipeline_batch(config)
    result = forward_numpy(init_params(config), batch, labels, config)
    print("batch_shape=", batch.shape)
    print("labels=", labels.tolist())
    print(f"loss={result['loss']:.6f}")


if __name__ == "__main__":
    main()
