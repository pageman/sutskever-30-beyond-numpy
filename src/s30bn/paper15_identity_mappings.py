"""Compact pre-activation residual CNN for paper 15."""

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
class IdentityResNetConfig:
    image_size: int = 12
    channels: int = 4
    num_classes: int = 4
    learning_rate: float = 0.08


def synthetic_identity_dataset(config: IdentityResNetConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = config.image_size
    xs = []
    ys = []
    for label in range(config.num_classes):
        image = np.zeros((config.channels, n, n), dtype=np.float64)
        if label == 0:
            image[0, :, ::2] = 1.0
            image[1, n // 2 - 1 : n // 2 + 1, :] = 0.5
        elif label == 1:
            image[1, ::2, :] = 1.0
            image[2, :, n // 2 - 1 : n // 2 + 1] = 0.5
        elif label == 2:
            np.fill_diagonal(image[0], 1.0)
            np.fill_diagonal(np.fliplr(image[3]), 1.0)
        else:
            image[:, n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 0.8
            image[0, :, :] += 0.1
        xs.append(np.clip(image, 0.0, 1.0))
        ys.append(label)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def init_params(config: IdentityResNetConfig, seed: int = 15) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.09
    return {
        "conv1_w": rng.normal(scale=scale, size=(config.channels, config.channels, 3, 3)),
        "conv1_b": np.zeros((config.channels,), dtype=np.float64),
        "conv2_w": rng.normal(scale=scale, size=(config.channels, config.channels, 3, 3)),
        "conv2_b": np.zeros((config.channels,), dtype=np.float64),
        "fc_w": rng.normal(scale=scale, size=(config.num_classes, config.channels)),
        "fc_b": np.zeros((config.num_classes,), dtype=np.float64),
    }


def zero_residual_params(params: ArrayDict) -> ArrayDict:
    updated = {key: value.copy() for key, value in params.items()}
    for key in ("conv1_w", "conv1_b", "conv2_w", "conv2_b"):
        updated[key].fill(0.0)
    return updated


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def identity_features_numpy(images: np.ndarray) -> np.ndarray:
    return np.maximum(images, 0.0)


def forward_tinygrad(
    params: Dict[str, Tensor],
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Tensor, Tensor]:
    x = Tensor(images, dtype=params["conv1_w"].dtype)
    preact = x.relu()
    residual = preact.conv2d(params["conv1_w"], bias=params["conv1_b"], padding=1).relu()
    residual = residual.conv2d(params["conv2_w"], bias=params["conv2_b"], padding=1)
    block = x + residual
    pooled = block.relu().mean(axis=(2, 3))
    logits = pooled @ params["fc_w"].transpose() + params["fc_b"]
    loss = logits.sparse_categorical_crossentropy(Tensor(labels.tolist()))
    return loss, logits


def forward_torch(
    params: Dict[str, torch.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(images, dtype=torch.float64)
    preact = F.relu(x)
    residual = F.relu(F.conv2d(preact, params["conv1_w"], bias=params["conv1_b"], padding=1))
    residual = F.conv2d(residual, params["conv2_w"], bias=params["conv2_b"], padding=1)
    block = x + residual
    pooled = F.relu(block).mean(dim=(2, 3))
    logits = pooled @ params["fc_w"].t() + params["fc_b"]
    loss = F.cross_entropy(logits, torch.tensor(labels))
    return loss, logits


def train_step_torch(
    params: Dict[str, torch.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> float:
    loss, _ = forward_torch(params, images, labels)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    images: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> float:
    loss, _ = forward_tinygrad(params, images, labels)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(images, dtype=jnp.float64)
    dims = ("NCHW", "OIHW", "NCHW")
    preact = jax.nn.relu(x)
    residual = jax.lax.conv_general_dilated(preact, params["conv1_w"], (1, 1), "SAME", dimension_numbers=dims)
    residual = jax.nn.relu(residual + params["conv1_b"][None, :, None, None])
    residual = jax.lax.conv_general_dilated(residual, params["conv2_w"], (1, 1), "SAME", dimension_numbers=dims)
    residual = residual + params["conv2_b"][None, :, None, None]
    block = x + residual
    pooled = jnp.mean(jax.nn.relu(block), axis=(2, 3))
    logits = pooled @ params["fc_w"].T + params["fc_b"]
    labels_j = jnp.asarray(labels)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, images, labels)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
