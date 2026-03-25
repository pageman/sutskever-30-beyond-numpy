"""Minimal NumPy sanity checks for paper 17."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper17_vae import VAEConfig, forward_numpy, init_params, synthetic_binary_data


def main() -> None:
    config = VAEConfig()
    data = synthetic_binary_data()
    result = forward_numpy(init_params(config), data, config)
    print("data_shape=", data.shape)
    print(f"loss={result['loss']:.6f}")
    print("mu_shape=", result["mu"].shape)


if __name__ == "__main__":
    main()
