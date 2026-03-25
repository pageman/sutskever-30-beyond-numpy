"""Tiny closed-form CTC example for paper 21."""

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
class CTCConfig:
    input_dim: int = 2
    num_classes_with_blank: int = 3
    learning_rate: float = 0.12
    target_label: int = 1
    blank_label: int = 0


def synthetic_ctc_batch() -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [[1.0, 0.0], [0.3, 0.9]],
            [[0.8, 0.2], [0.1, 1.0]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([1, 1], dtype=np.int64)
    return features, targets


def init_params(config: CTCConfig, seed: int = 21) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.25
    return {
        "W": rng.normal(scale=scale, size=(config.num_classes_with_blank, config.input_dim)),
        "b": np.zeros((config.num_classes_with_blank,), dtype=np.float64),
    }


def ctc_single_probability(probs: np.ndarray, target_label: int, blank_label: int) -> float:
    p0b = probs[0, blank_label]
    p0t = probs[0, target_label]
    p1b = probs[1, blank_label]
    p1t = probs[1, target_label]
    return float(p0b * p1t + p0t * p1b + p0t * p1t)


def forward_numpy(
    params: ArrayDict,
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> Dict[str, np.ndarray | float]:
    logits = features @ params["W"].T + params["b"]
    shifted = logits - logits.max(axis=2, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=2, keepdims=True)
    losses = []
    for sample_probs, target in zip(probs, targets):
        prob = ctc_single_probability(sample_probs, int(target), config.blank_label)
        losses.append(-np.log(prob + 1e-12))
    return {"loss": float(np.mean(losses)), "logits": logits}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def _ctc_single_probability_tinygrad(probs: Tensor, target_label: int, blank_label: int) -> Tensor:
    p0b = probs[0, blank_label]
    p0t = probs[0, target_label]
    p1b = probs[1, blank_label]
    p1t = probs[1, target_label]
    return p0b * p1t + p0t * p1b + p0t * p1t


def forward_tinygrad(
    params: Dict[str, Tensor],
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> Tuple[Tensor, Tensor]:
    x = Tensor(features.tolist(), dtype=params["W"].dtype)
    logits = x @ params["W"].transpose() + params["b"]
    probs = logits.softmax(axis=2)
    losses = []
    for i, target in enumerate(targets.tolist()):
        prob = _ctc_single_probability_tinygrad(probs[i], int(target), config.blank_label)
        losses.append(-(prob + 1e-12).log())
    loss = losses[0].stack(*losses[1:]).mean()
    return loss, logits


def forward_torch(
    params: Dict[str, torch.Tensor],
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(features, dtype=torch.float64)
    logits = x @ params["W"].t() + params["b"]
    log_probs = torch.log_softmax(logits, dim=2).transpose(0, 1)
    target_tensor = torch.tensor(targets, dtype=torch.long).unsqueeze(1)
    input_lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.long)
    target_lengths = torch.ones((features.shape[0],), dtype=torch.long)
    loss = F.ctc_loss(log_probs, target_tensor, input_lengths, target_lengths, blank=config.blank_label, reduction="mean", zero_infinity=True)
    return loss, logits


def train_step_torch(
    params: Dict[str, torch.Tensor],
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> float:
    loss, _ = forward_torch(params, features, targets, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> float:
    loss, _ = forward_tinygrad(params, features, targets, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(features, dtype=jnp.float64)
    logits = x @ params["W"].T + params["b"]
    probs = jax.nn.softmax(logits, axis=2)

    def single_loss(sample_probs: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        p0b = sample_probs[0, config.blank_label]
        p0t = sample_probs[0, target]
        p1b = sample_probs[1, config.blank_label]
        p1t = sample_probs[1, target]
        prob = p0b * p1t + p0t * p1b + p0t * p1t
        return -jnp.log(prob + 1e-12)

    losses = jax.vmap(single_loss)(probs, jnp.asarray(targets))
    return jnp.mean(losses), logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    features: np.ndarray,
    targets: np.ndarray,
    config: CTCConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, features, targets, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
