"""PyTorch implementation wrapper for paper 14."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper14_bahdanau_attention import (
    BahdanauConfig,
    forward_torch,
    init_params,
    params_to_torch,
    sample_pair,
)


def main() -> None:
    config = BahdanauConfig()
    source, decoder_token, target = sample_pair(config)
    params = params_to_torch(init_params(config))
    loss, logits, attention = forward_torch(params, source, decoder_token, target, config)
    print(f"torch loss: {loss.item():.6f}")
    print("logits shape:", tuple(logits.shape))
    print("attention shape:", tuple(attention.shape))


if __name__ == "__main__":
    main()
