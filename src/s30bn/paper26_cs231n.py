"""Tiny CNN fundamentals example for paper 26."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from s30bn.common import ArrayDict, seeded_rng

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CNNConfig:
    image_size: int = 8
    in_channels: int = 3
    hidden_channels: int = 4
    num_classes: int = 4
    learning_rate: float = 0.1


def synthetic_cifar_like(config: CNNConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = config.image_size
    xs = []
    ys = []
    for label in range(config.num_classes):
        image = np.zeros((config.in_channels, n, n), dtype=np.float64)
        if label == 0:
            image[:, :, ::2] = 1.0
        elif label == 1:
            image[:, ::2, :] = 1.0
        elif label == 2:
            np.fill_diagonal(image[0], 1.0)
            np.fill_diagonal(np.fliplr(image[1]), 1.0)
        else:
            image[:, n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1.0
        xs.append(image)
        ys.append(label)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def init_params(config: CNNConfig, seed: int = 2) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.15
    return {
        "conv_w": rng.normal(scale=scale, size=(config.hidden_channels, config.in_channels, 3, 3)),
        "conv_b": np.zeros((config.hidden_channels,), dtype=np.float64),
        "fc_w": rng.normal(scale=scale, size=(config.num_classes, config.hidden_channels)),
        "fc_b": np.zeros((config.num_classes,), dtype=np.float64),
    }


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def forward_torch(
    params: Dict[str, torch.Tensor],
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(images, dtype=torch.float64)
    conv = F.conv2d(x, params["conv_w"], bias=params["conv_b"], padding=1)
    act = F.relu(conv)
    pooled = act.mean(dim=(2, 3))
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


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_jax(
    params: Dict[str, jnp.ndarray],
    images: np.ndarray,
    labels: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(images, dtype=jnp.float64)
    dimension_numbers = ("NCHW", "OIHW", "NCHW")
    conv = jax.lax.conv_general_dilated(
        x,
        params["conv_w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=dimension_numbers,
    )
    conv = conv + params["conv_b"][None, :, None, None]
    act = jax.nn.relu(conv)
    pooled = jnp.mean(act, axis=(2, 3))
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
