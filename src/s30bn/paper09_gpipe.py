"""Tiny microbatch pipeline demo for paper 09."""

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
class GPipeConfig:
    input_dim: int = 4
    hidden_dim: int = 5
    output_dim: int = 3
    batch_size: int = 4
    microbatch_size: int = 2
    learning_rate: float = 0.08


def synthetic_pipeline_batch(config: GPipeConfig) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 1, 2, 1], dtype=np.int64)
    return x, y


def init_params(config: GPipeConfig, seed: int = 9) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W1": rng.normal(scale=scale, size=(config.hidden_dim, config.input_dim)),
        "b1": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W2": rng.normal(scale=scale, size=(config.output_dim, config.hidden_dim)),
        "b2": np.zeros((config.output_dim,), dtype=np.float64),
    }


def _numpy_stage1(params: ArrayDict, batch: np.ndarray) -> np.ndarray:
    return np.tanh(batch @ params["W1"].T + params["b1"])


def _numpy_stage2(params: ArrayDict, hidden: np.ndarray) -> np.ndarray:
    return hidden @ params["W2"].T + params["b2"]


def forward_numpy(
    params: ArrayDict,
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> Dict[str, np.ndarray | float]:
    logits_parts = []
    for start in range(0, batch.shape[0], config.microbatch_size):
        micro = batch[start : start + config.microbatch_size]
        hidden = _numpy_stage1(params, micro)
        logits_parts.append(_numpy_stage2(params, hidden))
    logits = np.concatenate(logits_parts, axis=0)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(probs[np.arange(labels.shape[0]), labels] + 1e-12)))
    return {"loss": loss, "logits": logits}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> Tuple[Tensor, Tensor]:
    logits_parts = []
    for start in range(0, batch.shape[0], config.microbatch_size):
        micro = Tensor(batch[start : start + config.microbatch_size].tolist(), dtype=params["W1"].dtype)
        hidden = (micro @ params["W1"].transpose() + params["b1"]).tanh()
        logits_parts.append(hidden @ params["W2"].transpose() + params["b2"])
    logits = logits_parts[0].cat(*logits_parts[1:], dim=0)
    loss = logits.sparse_categorical_crossentropy(Tensor(labels.tolist()))
    return loss, logits


def forward_torch(
    params: Dict[str, torch.Tensor],
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_parts = []
    for start in range(0, batch.shape[0], config.microbatch_size):
        micro = torch.tensor(batch[start : start + config.microbatch_size], dtype=torch.float64)
        hidden = torch.tanh(micro @ params["W1"].t() + params["b1"])
        logits_parts.append(hidden @ params["W2"].t() + params["b2"])
    logits = torch.cat(logits_parts, dim=0)
    loss = F.cross_entropy(logits, torch.tensor(labels))
    return loss, logits


def train_step_torch(
    params: Dict[str, torch.Tensor],
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> float:
    loss, _ = forward_torch(params, batch, labels, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> float:
    loss, _ = forward_tinygrad(params, batch, labels, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(batch, dtype=jnp.float64)
    logits_parts = []
    for start in range(0, batch.shape[0], config.microbatch_size):
        micro = x[start : start + config.microbatch_size]
        hidden = jnp.tanh(micro @ params["W1"].T + params["b1"])
        logits_parts.append(hidden @ params["W2"].T + params["b2"])
    logits = jnp.concatenate(logits_parts, axis=0)
    labels_j = jnp.asarray(labels)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    batch: np.ndarray,
    labels: np.ndarray,
    config: GPipeConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, batch, labels, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
