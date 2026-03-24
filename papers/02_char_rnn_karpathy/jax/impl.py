"""JAX implementation wrapper for paper 02."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper02_char_rnn import CharRNNConfig, build_dataset, forward_jax, init_params, params_to_jax


def main() -> None:
    config = CharRNNConfig()
    inputs, targets = build_dataset(config)
    params = params_to_jax(init_params(config))
    loss, logits = forward_jax(params, inputs, targets)
    print(f"jax loss: {float(loss):.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
