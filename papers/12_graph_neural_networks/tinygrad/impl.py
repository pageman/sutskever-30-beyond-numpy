"""tinygrad implementation wrapper for paper 12."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper12_gnn import GNNConfig, forward_tinygrad, init_params, params_to_tinygrad, synthetic_graph_dataset


def main() -> None:
    config = GNNConfig()
    graphs, adjacency, labels = synthetic_graph_dataset(config)
    params = params_to_tinygrad(init_params(config))
    loss, logits = forward_tinygrad(params, graphs, adjacency, labels)
    print(f"tinygrad loss: {loss.item():.6f}")
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()
