"""JAX implementation wrapper for paper 15."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper15_identity_mappings import (
    IdentityResNetConfig,
    forward_jax,
    init_params,
    params_to_jax,
    synthetic_identity_dataset,
)


def main() -> None:
    config = IdentityResNetConfig()
    images, labels = synthetic_identity_dataset(config)
    params = params_to_jax(init_params(config))
    loss, logits = forward_jax(params, images, labels)
    print(f"jax loss: {float(loss):.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
