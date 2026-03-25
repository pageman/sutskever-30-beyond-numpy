"""Minimal NumPy sanity checks for paper 12."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper12_gnn import GNNConfig, init_params, synthetic_graph_dataset


def main() -> None:
    config = GNNConfig()
    graphs, adjacency, labels = synthetic_graph_dataset(config)
    params = init_params(config)
    print("graphs_shape=", graphs.shape)
    print("adjacency_shape=", adjacency.shape)
    print("labels=", labels.tolist())
    print("message_weight_shape=", params["W_msg"].shape)


if __name__ == "__main__":
    main()
