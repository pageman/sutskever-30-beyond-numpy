"""Tiny positional-retrieval model for paper 30."""

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
class LostMiddleConfig:
    seq_len: int = 5
    chunk_dim: int = 2
    hidden_dim: int = 3
    num_classes: int = 2
    learning_rate: float = 0.04


def synthetic_middle_batch() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = np.asarray(
        [
            [[1.0, 0.0], [0.1, 0.0], [0.0, 1.0], [0.1, 0.0], [0.8, 0.0]],
            [[0.0, 1.0], [0.0, 0.2], [1.0, 0.0], [0.0, 0.2], [0.0, 0.8]],
            [[0.9, 0.0], [0.1, 0.0], [0.0, 1.0], [0.1, 0.0], [0.7, 0.0]],
            [[0.0, 0.9], [0.0, 0.1], [1.0, 0.0], [0.0, 0.1], [0.0, 0.7]],
        ],
        dtype=np.float64,
    )
    queries = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return chunks, queries, targets


def init_params(config: LostMiddleConfig, seed: int = 30) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_chunk": rng.normal(scale=scale, size=(config.hidden_dim, config.chunk_dim)),
        "b_chunk": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_query": rng.normal(scale=scale, size=(config.hidden_dim, config.chunk_dim)),
        "b_query": np.zeros((config.hidden_dim,), dtype=np.float64),
        "position_bias": np.asarray([0.35, 0.05, -0.35, 0.05, 0.35], dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.num_classes, config.hidden_dim)),
        "b_out": np.zeros((config.num_classes,), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    chunk_h = np.tanh(chunks @ params["W_chunk"].T + params["b_chunk"])
    query_h = np.tanh(queries @ params["W_query"].T + params["b_query"])
    scores = np.einsum("bsh,bh->bs", chunk_h, query_h) + params["position_bias"][None, :]
    weights = stable_softmax(scores, axis=1)
    pooled = np.einsum("bs,bsh->bh", weights, chunk_h)
    logits = pooled @ params["W_out"].T + params["b_out"]
    probs = stable_softmax(logits, axis=1)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "logits": logits, "weights": weights}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    logits_parts = []
    for chunk_seq, query in zip(chunks, queries):
        chunk_h = []
        for chunk in chunk_seq:
            chunk_h.append((params["W_chunk"] @ Tensor(chunk.tolist(), dtype=params["W_chunk"].dtype) + params["b_chunk"]).tanh())
        query_h = (params["W_query"] @ Tensor(query.tolist(), dtype=params["W_query"].dtype) + params["b_query"]).tanh()
        scores = []
        for idx, h in enumerate(chunk_h):
            scores.append((h * query_h).sum() + params["position_bias"][idx])
        weights = scores[0].stack(*scores[1:]).softmax(axis=0)
        pooled = (weights.reshape(-1, 1) * chunk_h[0].stack(*chunk_h[1:])).sum(axis=0)
        logits_parts.append(params["W_out"] @ pooled + params["b_out"])
    logits = logits_parts[0].stack(*logits_parts[1:])
    loss = logits.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits


def forward_torch(params: Dict[str, torch.Tensor], chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    chunk_h = torch.tanh(torch.tensor(chunks, dtype=torch.float64) @ params["W_chunk"].t() + params["b_chunk"])
    query_h = torch.tanh(torch.tensor(queries, dtype=torch.float64) @ params["W_query"].t() + params["b_query"])
    scores = torch.einsum("bsh,bh->bs", chunk_h, query_h) + params["position_bias"].unsqueeze(0)
    weights = torch.softmax(scores, dim=1)
    pooled = torch.einsum("bs,bsh->bh", weights, chunk_h)
    logits = pooled @ params["W_out"].t() + params["b_out"]
    loss = F.cross_entropy(logits, torch.tensor(targets))
    return loss, logits


def train_step_torch(
    params: Dict[str, torch.Tensor], chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray, learning_rate: float
) -> float:
    loss, _ = forward_torch(params, chunks, queries, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor], chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray, learning_rate: float
) -> float:
    loss, _ = forward_tinygrad(params, chunks, queries, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray], chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    chunk_h = jnp.tanh(jnp.asarray(chunks, dtype=jnp.float64) @ params["W_chunk"].T + params["b_chunk"])
    query_h = jnp.tanh(jnp.asarray(queries, dtype=jnp.float64) @ params["W_query"].T + params["b_query"])
    scores = jnp.einsum("bsh,bh->bs", chunk_h, query_h) + params["position_bias"][None, :]
    weights = jax.nn.softmax(scores, axis=1)
    pooled = jnp.einsum("bs,bsh->bh", weights, chunk_h)
    logits = pooled @ params["W_out"].T + params["b_out"]
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray], chunks: np.ndarray, queries: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, chunks, queries, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
