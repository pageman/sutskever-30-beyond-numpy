"""Compact dilated-convolution classifier for paper 11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("LLVM", "1")
from tinygrad import Tensor

from s30bn.common import ArrayDict, seeded_rng

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class DilatedConvConfig:
    image_size: int = 12
    in_channels: int = 2
    hidden_channels: int = 4
    out_channels: int = 4
    num_classes: int = 4
    learning_rate: float = 0.08
    dilation: int = 2


def synthetic_context_dataset(config: DilatedConvConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = config.image_size
    xs = []
    ys = []
    for label in range(config.num_classes):
        image = np.zeros((config.in_channels, n, n), dtype=np.float64)
        if label == 0:
            image[0, :, ::3] = 1.0
            image[1, 1::3, :] = 0.4
        elif label == 1:
            image[0, ::3, :] = 1.0
            image[1, :, 1::3] = 0.4
        elif label == 2:
            image[0, 2:n - 2, 2:n - 2] = 0.5
            image[1, ::2, ::2] = 1.0
        else:
            image[:, n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1.0
        xs.append(image)
        ys.append(label)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def dilate_kernel(kernel: np.ndarray, dilation: int) -> np.ndarray:
    kh, kw = kernel.shape[-2:]
    out_h = kh + (kh - 1) * (dilation - 1)
    out_w = kw + (kw - 1) * (dilation - 1)
    expanded = np.zeros((*kernel.shape[:-2], out_h, out_w), dtype=kernel.dtype)
    expanded[..., ::dilation, ::dilation] = kernel
    return expanded


def init_params(config: DilatedConvConfig, seed: int = 11) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.11
    base_dilated = rng.normal(scale=scale, size=(config.out_channels, config.hidden_channels, 3, 3))
    return {
        "conv1_w": rng.normal(scale=scale, size=(config.hidden_channels, config.in_channels, 3, 3)),
        "conv1_b": np.zeros((config.hidden_channels,), dtype=np.float64),
        "dilated_w": dilate_kernel(base_dilated, config.dilation),
        "dilated_b": np.zeros((config.out_channels,), dtype=np.float64),
        "fc_w": rng.normal(scale=scale, size=(config.num_classes, config.out_channels)),
        "fc_b": np.zeros((config.num_classes,), dtype=np.float64),
    }


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    config: DilatedConvConfig,
) -> Tuple[Tensor, Tensor]:
    x = Tensor(images, dtype=params["conv1_w"].dtype)
    hidden = x.conv2d(params["conv1_w"], bias=params["conv1_b"], padding=1).relu()
    features = hidden.conv2d(params["dilated_w"], bias=params["dilated_b"], padding=2).relu()
    pooled = features.mean(axis=(2, 3))
    logits = pooled @ params["fc_w"].transpose() + params["fc_b"]
    loss = logits.sparse_categorical_crossentropy(Tensor(labels.tolist()))
    return loss, logits


def forward_torch(
    params: Dict[str, torch.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    config: DilatedConvConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(images, dtype=torch.float64)
    hidden = F.relu(F.conv2d(x, params["conv1_w"], bias=params["conv1_b"], padding=1))
    features = F.relu(F.conv2d(hidden, params["dilated_w"], bias=params["dilated_b"], padding=2))
    pooled = features.mean(dim=(2, 3))
    logits = pooled @ params["fc_w"].t() + params["fc_b"]
    loss = F.cross_entropy(logits, torch.tensor(labels))
    return loss, logits


def train_step_torch(
    params: Dict[str, torch.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    config: DilatedConvConfig,
) -> float:
    loss, _ = forward_torch(params, images, labels, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    config: DilatedConvConfig,
) -> float:
    loss, _ = forward_tinygrad(params, images, labels, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
    config: DilatedConvConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(images, dtype=jnp.float64)
    dims = ("NCHW", "OIHW", "NCHW")
    hidden = jax.lax.conv_general_dilated(x, params["conv1_w"], (1, 1), "SAME", dimension_numbers=dims)
    hidden = jax.nn.relu(hidden + params["conv1_b"][None, :, None, None])
    features = jax.lax.conv_general_dilated(hidden, params["dilated_w"], (1, 1), "SAME", dimension_numbers=dims)
    features = jax.nn.relu(features + params["dilated_b"][None, :, None, None])
    pooled = jnp.mean(features, axis=(2, 3))
    logits = pooled @ params["fc_w"].T + params["fc_b"]
    labels_j = jnp.asarray(labels)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
    config: DilatedConvConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, images, labels, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
