"""JAX implementation wrapper for paper 08."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper08_seq2seq_sets import SetSeqConfig, forward_jax, init_params, params_to_jax, synthetic_set_batch


def main() -> None:
    config = SetSeqConfig()
    sets, targets = synthetic_set_batch()
    params = params_to_jax(init_params(config))
    loss, logits = forward_jax(params, sets, targets)
    print(f"jax loss: {float(loss):.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
