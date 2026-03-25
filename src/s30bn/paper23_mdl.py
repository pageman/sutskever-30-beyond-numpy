"""Tiny MDL-style classifier for paper 23."""

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

from s30bn.common import ArrayDict, seeded_rng, stable_softmax

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MDLConfig:
    input_dim: int = 3
    num_classes: int = 2
    complexity_weight: float = 0.05
    learning_rate: float = 0.08


def synthetic_mdl_batch() -> Tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, 0.2],
            [0.9, 0.1, 0.1],
            [0.1, 0.9, 0.1],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return inputs, targets


def init_params(config: MDLConfig, seed: int = 23) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W": rng.normal(scale=scale, size=(config.num_classes, config.input_dim)),
        "b": np.zeros((config.num_classes,), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, inputs: np.ndarray, targets: np.ndarray, config: MDLConfig) -> Dict[str, np.ndarray | float]:
    logits = inputs @ params["W"].T + params["b"]
    probs = stable_softmax(logits, axis=1)
    empirical = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    complexity = float(config.complexity_weight * (np.mean(params["W"] ** 2) + np.mean(params["b"] ** 2)))
    return {"loss": empirical + complexity, "logits": logits}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], inputs: np.ndarray, targets: np.ndarray, config: MDLConfig) -> Tuple[Tensor, Tensor]:
    x = Tensor(inputs.tolist(), dtype=params["W"].dtype)
    logits = x @ params["W"].transpose() + params["b"]
    ce = logits.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    complexity = config.complexity_weight * (((params["W"] * params["W"]).mean()) + ((params["b"] * params["b"]).mean()))
    return ce + complexity, logits


def forward_torch(
    params: Dict[str, torch.Tensor], inputs: np.ndarray, targets: np.ndarray, config: MDLConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(inputs, dtype=torch.float64)
    logits = x @ params["W"].t() + params["b"]
    ce = F.cross_entropy(logits, torch.tensor(targets))
    complexity = config.complexity_weight * (torch.mean(params["W"] ** 2) + torch.mean(params["b"] ** 2))
    return ce + complexity, logits


def train_step_torch(
    params: Dict[str, torch.Tensor], inputs: np.ndarray, targets: np.ndarray, config: MDLConfig
) -> float:
    loss, _ = forward_torch(params, inputs, targets, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor], inputs: np.ndarray, targets: np.ndarray, config: MDLConfig
) -> float:
    loss, _ = forward_tinygrad(params, inputs, targets, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray], inputs: np.ndarray, targets: np.ndarray, config: MDLConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(inputs, dtype=jnp.float64)
    logits = x @ params["W"].T + params["b"]
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    empirical = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    complexity = config.complexity_weight * (jnp.mean(params["W"] ** 2) + jnp.mean(params["b"] ** 2))
    return empirical + complexity, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray], inputs: np.ndarray, targets: np.ndarray, config: MDLConfig
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, inputs, targets, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
