"""Minimal NumPy sanity checks for paper 08."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper08_seq2seq_sets import SetSeqConfig, forward_numpy, init_params, synthetic_set_batch


def main() -> None:
    config = SetSeqConfig()
    sets, targets = synthetic_set_batch()
    result = forward_numpy(init_params(config), sets, targets)
    print("sets_shape=", sets.shape)
    print("targets=", targets.tolist())
    print(f"loss={result['loss']:.6f}")


if __name__ == "__main__":
    main()
