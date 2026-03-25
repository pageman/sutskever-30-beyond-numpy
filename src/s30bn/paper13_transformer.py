"""Tiny single-head self-attention model for paper 13."""

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
class TransformerConfig:
    seq_len: int = 3
    model_dim: int = 4
    num_classes: int = 3
    learning_rate: float = 0.08


def synthetic_attention_batch() -> Tuple[np.ndarray, np.ndarray]:
    seqs = np.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.8, 0.1, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.1, 0.8, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.1, 0.0, 0.0], [0.9, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0], [1.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return seqs, targets


def init_params(config: TransformerConfig, seed: int = 13) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    d = config.model_dim
    return {
        "Wq": rng.normal(scale=scale, size=(d, d)),
        "Wk": rng.normal(scale=scale, size=(d, d)),
        "Wv": rng.normal(scale=scale, size=(d, d)),
        "Wo": rng.normal(scale=scale, size=(config.num_classes, d)),
        "bo": np.zeros((config.num_classes,), dtype=np.float64),
    }


def attention_weights_numpy(params: ArrayDict, seqs: np.ndarray) -> np.ndarray:
    q = seqs @ params["Wq"].T
    k = seqs @ params["Wk"].T
    scores = np.matmul(q, np.swapaxes(k, 1, 2)) / np.sqrt(seqs.shape[2])
    return stable_softmax(scores, axis=2)


def forward_numpy(params: ArrayDict, seqs: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    q = seqs @ params["Wq"].T
    k = seqs @ params["Wk"].T
    v = seqs @ params["Wv"].T
    weights = attention_weights_numpy(params, seqs)
    context = np.matmul(weights, v)
    pooled = np.mean(context, axis=1)
    logits = pooled @ params["Wo"].T + params["bo"]
    probs = stable_softmax(logits, axis=1)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "logits": logits, "weights": weights}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], seqs: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    logits_parts = []
    scale = float(np.sqrt(seqs.shape[2]))
    for seq in seqs:
        x = Tensor(seq.tolist(), dtype=params["Wq"].dtype)
        q = x @ params["Wq"].transpose()
        k = x @ params["Wk"].transpose()
        v = x @ params["Wv"].transpose()
        scores = (q @ k.transpose()) / scale
        weights = scores.softmax(axis=1)
        context = weights @ v
        pooled = context.mean(axis=0)
        logits_parts.append(params["Wo"] @ pooled + params["bo"])
    logits = logits_parts[0].stack(*logits_parts[1:])
    loss = logits.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits


def forward_torch(params: Dict[str, torch.Tensor], seqs: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(seqs, dtype=torch.float64)
    q = x @ params["Wq"].t()
    k = x @ params["Wk"].t()
    v = x @ params["Wv"].t()
    scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(seqs.shape[2])
    weights = torch.softmax(scores, dim=2)
    context = torch.matmul(weights, v)
    pooled = context.mean(dim=1)
    logits = pooled @ params["Wo"].t() + params["bo"]
    loss = F.cross_entropy(logits, torch.tensor(targets))
    return loss, logits


def train_step_torch(params: Dict[str, torch.Tensor], seqs: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_torch(params, seqs, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(params: Dict[str, Tensor], seqs: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_tinygrad(params, seqs, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(params: Dict[str, jnp.ndarray], seqs: np.ndarray, targets: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(seqs, dtype=jnp.float64)
    q = x @ params["Wq"].T
    k = x @ params["Wk"].T
    v = x @ params["Wv"].T
    scores = jnp.matmul(q, jnp.swapaxes(k, 1, 2)) / jnp.sqrt(seqs.shape[2])
    weights = jax.nn.softmax(scores, axis=2)
    context = jnp.matmul(weights, v)
    pooled = jnp.mean(context, axis=1)
    logits = pooled @ params["Wo"].T + params["bo"]
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray], seqs: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, seqs, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
