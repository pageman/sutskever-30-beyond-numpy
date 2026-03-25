"""Minimal NumPy sanity checks for paper 16."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper16_relational_reasoning import RelationConfig, init_params, synthetic_relation_dataset


def main() -> None:
    config = RelationConfig()
    scenes, labels = synthetic_relation_dataset(config)
    params = init_params(config)
    print("scenes_shape=", scenes.shape)
    print("labels=", labels.tolist())
    print("pair_hidden_shape=", params["g_w1"].shape)


if __name__ == "__main__":
    main()
