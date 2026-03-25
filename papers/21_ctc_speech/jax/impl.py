"""JAX implementation wrapper for paper 21."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper21_ctc import CTCConfig, forward_jax, init_params, params_to_jax, synthetic_ctc_batch


def main() -> None:
    config = CTCConfig()
    features, targets = synthetic_ctc_batch()
    params = params_to_jax(init_params(config))
    loss, logits = forward_jax(params, features, targets, config)
    print(f"jax loss: {float(loss):.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
