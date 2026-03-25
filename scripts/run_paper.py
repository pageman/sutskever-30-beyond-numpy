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
from s30bn.paper04_rnn_regularization import (
    RNNRegularizationConfig,
    build_dataset as build_rnnreg_dataset,
    forward_numpy as rnnreg_forward_numpy,
    init_params as init_rnnreg_params,
)
from s30bn.paper07_alexnet import AlexNetConfig, init_params as init_alexnet_params, synthetic_imagenet_like
from s30bn.paper10_resnet import ResNetConfig, init_params as init_resnet_params, synthetic_residual_dataset
from s30bn.paper15_identity_mappings import (
    IdentityResNetConfig,
    init_params as init_identity_params,
    synthetic_identity_dataset,
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


def run_04() -> None:
    config = RNNRegularizationConfig()
    inputs, targets = build_rnnreg_dataset(config)
    result = rnnreg_forward_numpy(init_rnnreg_params(config), inputs, targets, config)
    print(f"paper 04 numpy loss: {result['loss']:.6f}")


def run_07() -> None:
    config = AlexNetConfig()
    images, labels = synthetic_imagenet_like(config)
    params = init_alexnet_params(config)
    print(f"paper 07 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 07 conv1 kernel shape: {params['conv1_w'].shape}")


def run_10() -> None:
    config = ResNetConfig()
    images, labels = synthetic_residual_dataset(config)
    params = init_resnet_params(config)
    print(f"paper 10 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 10 stem kernel shape: {params['stem_w'].shape}")


def run_15() -> None:
    config = IdentityResNetConfig()
    images, labels = synthetic_identity_dataset(config)
    params = init_identity_params(config)
    print(f"paper 15 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 15 residual kernel shape: {params['conv1_w'].shape}")


def run_26() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = init_cnn_params(config)
    print(f"paper 26 synthetic dataset shape: {images.shape}, labels: {labels.tolist()}")
    print(f"paper 26 conv kernel shape: {params['conv_w'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True, choices=["02", "03", "04", "07", "10", "15", "26"])
    args = parser.parse_args()
    {
        "02": run_02,
        "03": run_03,
        "04": run_04,
        "07": run_07,
        "10": run_10,
        "15": run_15,
        "26": run_26,
    }[args.paper]()


if __name__ == "__main__":
    main()
