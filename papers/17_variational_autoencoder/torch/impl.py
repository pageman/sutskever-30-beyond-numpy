"""PyTorch implementation wrapper for paper 17."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper17_vae import VAEConfig, forward_torch, init_params, params_to_torch, synthetic_binary_data


def main() -> None:
    config = VAEConfig()
    data = synthetic_binary_data()
    params = params_to_torch(init_params(config))
    loss, logits = forward_torch(params, data, config)
    print(f"torch loss: {loss.item():.6f}")
    print("logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
