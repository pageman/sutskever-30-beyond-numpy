"""Tiny dense passage retrieval model for paper 28."""

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
class DPRConfig:
    input_dim: int = 3
    hidden_dim: int = 4
    learning_rate: float = 0.08


def synthetic_dpr_batch() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    passages = np.asarray(
        [
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
            [0.8, 0.2, 0.1],
            [0.2, 0.8, 0.1],
        ],
        dtype=np.float64,
    )
    targets = np.arange(queries.shape[0], dtype=np.int64)
    return queries, passages, targets


def init_params(config: DPRConfig, seed: int = 28) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_q": rng.normal(scale=scale, size=(config.hidden_dim, config.input_dim)),
        "b_q": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_p": rng.normal(scale=scale, size=(config.hidden_dim, config.input_dim)),
        "b_p": np.zeros((config.hidden_dim,), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, queries: np.ndarray, passages: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    q = np.tanh(queries @ params["W_q"].T + params["b_q"])
    p = np.tanh(passages @ params["W_p"].T + params["b_p"])
    scores = q @ p.T
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    loss = float(-np.mean(log_probs[np.arange(targets.shape[0]), targets]))
    return {"loss": loss, "scores": scores}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    q = []
    for query in queries:
        x = Tensor(query.tolist(), dtype=params["W_q"].dtype)
        q.append((params["W_q"] @ x + params["b_q"]).tanh())
    p = []
    for passage in passages:
        x = Tensor(passage.tolist(), dtype=params["W_p"].dtype)
        p.append((params["W_p"] @ x + params["b_p"]).tanh())
    rows = []
    for query_h in q:
        rows.append((query_h.reshape(1, -1) * p[0].stack(*p[1:])).sum(axis=1))
    scores = rows[0].stack(*rows[1:])
    loss = scores.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, scores


def forward_torch(params: Dict[str, torch.Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.tanh(torch.tensor(queries, dtype=torch.float64) @ params["W_q"].t() + params["b_q"])
    p = torch.tanh(torch.tensor(passages, dtype=torch.float64) @ params["W_p"].t() + params["b_p"])
    scores = q @ p.t()
    loss = F.cross_entropy(scores, torch.tensor(targets))
    return loss, scores


def train_step_torch(
    params: Dict[str, torch.Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray, learning_rate: float
) -> float:
    loss, _ = forward_torch(params, queries, passages, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray, learning_rate: float
) -> float:
    loss, _ = forward_tinygrad(params, queries, passages, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q = jnp.tanh(jnp.asarray(queries, dtype=jnp.float64) @ params["W_q"].T + params["b_q"])
    p = jnp.tanh(jnp.asarray(passages, dtype=jnp.float64) @ params["W_p"].T + params["b_p"])
    scores = q @ p.T
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(scores, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, scores


def train_step_jax(
    params: Dict[str, jnp.ndarray], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, queries, passages, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
