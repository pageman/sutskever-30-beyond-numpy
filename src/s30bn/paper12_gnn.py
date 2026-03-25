"""Tiny message-passing GNN for paper 12."""

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
class GNNConfig:
    num_nodes: int = 3
    node_dim: int = 2
    hidden_dim: int = 4
    num_classes: int = 2
    learning_rate: float = 0.08


def synthetic_graph_dataset(config: GNNConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    graphs = np.asarray(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            [[0.0, 1.0], [0.1, 0.9], [1.0, 0.0]],
            [[1.0, 0.2], [0.8, 0.1], [0.2, 1.0]],
            [[0.1, 1.0], [0.2, 0.9], [1.0, 0.1]],
        ],
        dtype=np.float64,
    )
    adjacency = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
    return graphs, adjacency, labels


def init_params(config: GNNConfig, seed: int = 12) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_msg": rng.normal(scale=scale, size=(config.hidden_dim, config.node_dim)),
        "b_msg": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_self": rng.normal(scale=scale, size=(config.hidden_dim, config.node_dim)),
        "b_self": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.num_classes, config.hidden_dim)),
        "b_out": np.zeros((config.num_classes,), dtype=np.float64),
    }


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    graphs: np.ndarray,
    adjacency: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Tensor, Tensor]:
    adj = Tensor(adjacency.tolist(), dtype=params["W_msg"].dtype)
    logits = []
    for graph in graphs:
        nodes = Tensor(graph.tolist(), dtype=params["W_msg"].dtype)
        messages = (nodes @ params["W_msg"].transpose() + params["b_msg"]).relu()
        self_term = (nodes @ params["W_self"].transpose() + params["b_self"]).relu()
        aggregated = adj @ messages + self_term
        pooled = aggregated.mean(axis=0)
        logits.append(params["W_out"] @ pooled + params["b_out"])
    logits_tensor = logits[0].stack(*logits[1:])
    loss = logits_tensor.sparse_categorical_crossentropy(Tensor(labels.tolist()))
    return loss, logits_tensor


def forward_torch(
    params: Dict[str, torch.Tensor],
    graphs: np.ndarray,
    adjacency: np.ndarray,
    labels: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    adj = torch.tensor(adjacency, dtype=torch.float64)
    logits = []
    for graph in graphs:
        nodes = torch.tensor(graph, dtype=torch.float64)
        messages = F.relu(nodes @ params["W_msg"].t() + params["b_msg"])
        self_term = F.relu(nodes @ params["W_self"].t() + params["b_self"])
        aggregated = adj @ messages + self_term
        pooled = aggregated.mean(dim=0)
        logits.append(params["W_out"] @ pooled + params["b_out"])
    logits_tensor = torch.stack(logits)
    loss = F.cross_entropy(logits_tensor, torch.tensor(labels))
    return loss, logits_tensor


def train_step_torch(
    params: Dict[str, torch.Tensor],
    graphs: np.ndarray,
    adjacency: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> float:
    loss, _ = forward_torch(params, graphs, adjacency, labels)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    graphs: np.ndarray,
    adjacency: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> float:
    loss, _ = forward_tinygrad(params, graphs, adjacency, labels)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    graphs: np.ndarray,
    adjacency: np.ndarray,
    labels: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    graphs_j = jnp.asarray(graphs, dtype=jnp.float64)
    adj = jnp.asarray(adjacency, dtype=jnp.float64)

    def graph_logits(graph: jnp.ndarray) -> jnp.ndarray:
        messages = jax.nn.relu(graph @ params["W_msg"].T + params["b_msg"])
        self_term = jax.nn.relu(graph @ params["W_self"].T + params["b_self"])
        aggregated = adj @ messages + self_term
        pooled = jnp.mean(aggregated, axis=0)
        return params["W_out"] @ pooled + params["b_out"]

    logits = jax.vmap(graph_logits)(graphs_j)
    labels_j = jnp.asarray(labels)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    graphs: np.ndarray,
    adjacency: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, graphs, adjacency, labels)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
