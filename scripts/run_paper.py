#!/usr/bin/env python3
"""Run tiny demos for the populated papers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s30bn.paper02_char_rnn import (
    CharRNNConfig,
    build_dataset as build_char_dataset,
    forward_numpy as char_forward_numpy,
    init_params as init_char_params,
)
from s30bn.paper03_lstm import (
    LSTMConfig,
    build_dataset as build_lstm_dataset,
    forward_numpy as lstm_forward_numpy,
    init_params as init_lstm_params,
)
from s30bn.paper26_cs231n import CNNConfig, init_params as init_cnn_params, synthetic_cifar_like


def run_02() -> None:
    config = CharRNNConfig()
    inputs, targets = build_char_dataset(config)
    result = char_forward_numpy(init_char_params(config), inputs, targets)
    print(f"paper 02 numpy loss: {result['loss']:.6f}")


def run_03() -> None:
    config = LSTMConfig()
    inputs, targets = build_lstm_dataset(config)
    result = lstm_forward_numpy(init_lstm_params(config), inputs, targets)
    print(f"paper 03 numpy loss: {result['loss']:.6f}")


def run_26() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = init_cnn_params(config)
    print(f"paper 26 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 26 conv kernel shape: {params['conv_w'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, choices=["02", "03", "26"])
    args = parser.parse_args()
    {"02": run_02, "03": run_03, "26": run_26}[args.paper]()


if __name__ == "__main__":
    main()
