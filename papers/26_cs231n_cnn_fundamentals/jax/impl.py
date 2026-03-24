"""JAX implementation wrapper for paper 26."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper26_cs231n import CNNConfig, forward_jax, init_params, params_to_jax, synthetic_cifar_like


def main() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = params_to_jax(init_params(config))
    loss, logits = forward_jax(params, images, labels)
    print(f"jax loss: {float(loss):.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
