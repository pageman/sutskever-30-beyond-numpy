"""Tiny relation network for paper 16."""

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
class RelationConfig:
    num_objects: int = 3
    object_dim: int = 3
    hidden_dim: int = 5
    num_classes: int = 2
    learning_rate: float = 0.08


def synthetic_relation_dataset(config: RelationConfig) -> Tuple[np.ndarray, np.ndarray]:
    scenes = np.asarray(
        [
            [[1.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.2, 0.0, 1.0], [1.0, 0.0, 0.1]],
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.1], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([1, 0, 1, 1], dtype=np.int64)
    return scenes, labels


def init_params(config: RelationConfig, seed: int = 16) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    pair_dim = 2 * config.object_dim
    return {
        "g_w1": rng.normal(scale=scale, size=(config.hidden_dim, pair_dim)),
        "g_b1": np.zeros((config.hidden_dim,), dtype=np.float64),
        "g_w2": rng.normal(scale=scale, size=(config.hidden_dim, config.hidden_dim)),
        "g_b2": np.zeros((config.hidden_dim,), dtype=np.float64),
        "f_w": rng.normal(scale=scale, size=(config.num_classes, config.hidden_dim)),
        "f_b": np.zeros((config.num_classes,), dtype=np.float64),
    }


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    scenes: np.ndarray,
    labels: np.ndarray,
    config: RelationConfig,
) -> Tuple[Tensor, Tensor]:
    logits = []
    for scene in scenes:
        relations = []
        for i in range(config.num_objects):
            for j in range(config.num_objects):
                pair = np.concatenate([scene[i], scene[j]])
                pair_t = Tensor(pair.tolist(), dtype=params["g_w1"].dtype)
                hidden = (params["g_w1"] @ pair_t + params["g_b1"]).relu()
                hidden = (params["g_w2"] @ hidden + params["g_b2"]).relu()
                relations.append(hidden)
        relation_tensor = relations[0].stack(*relations[1:]).sum(axis=0)
        logits.append(params["f_w"] @ relation_tensor + params["f_b"])
    logits_tensor = logits[0].stack(*logits[1:])
    loss = logits_tensor.sparse_categorical_crossentropy(Tensor(labels.tolist()))
    return loss, logits_tensor


def forward_torch(
    params: Dict[str, torch.Tensor],
    scenes: np.ndarray,
    labels: np.ndarray,
    config: RelationConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = []
    for scene in scenes:
        relations = []
        for i in range(config.num_objects):
            for j in range(config.num_objects):
                pair = torch.tensor(np.concatenate([scene[i], scene[j]]), dtype=torch.float64)
                hidden = F.relu(params["g_w1"] @ pair + params["g_b1"])
                hidden = F.relu(params["g_w2"] @ hidden + params["g_b2"])
                relations.append(hidden)
        relation_tensor = torch.stack(relations).sum(dim=0)
        logits.append(params["f_w"] @ relation_tensor + params["f_b"])
    logits_tensor = torch.stack(logits)
    loss = F.cross_entropy(logits_tensor, torch.tensor(labels))
    return loss, logits_tensor


def train_step_torch(
    params: Dict[str, torch.Tensor],
    scenes: np.ndarray,
    labels: np.ndarray,
    config: RelationConfig,
) -> float:
    loss, _ = forward_torch(params, scenes, labels, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    scenes: np.ndarray,
    labels: np.ndarray,
    config: RelationConfig,
) -> float:
    loss, _ = forward_tinygrad(params, scenes, labels, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    scenes: np.ndarray,
    labels: np.ndarray,
    config: RelationConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    scenes_j = jnp.asarray(scenes, dtype=jnp.float64)

    def scene_logits(scene: jnp.ndarray) -> jnp.ndarray:
        rels = []
        for i in range(config.num_objects):
            for j in range(config.num_objects):
                pair = jnp.concatenate([scene[i], scene[j]])
                hidden = jax.nn.relu(params["g_w1"] @ pair + params["g_b1"])
                hidden = jax.nn.relu(params["g_w2"] @ hidden + params["g_b2"])
                rels.append(hidden)
        relation_tensor = jnp.stack(rels).sum(axis=0)
        return params["f_w"] @ relation_tensor + params["f_b"]

    logits = jax.vmap(scene_logits)(scenes_j)
    labels_j = jnp.asarray(labels)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    scenes: np.ndarray,
    labels: np.ndarray,
    config: RelationConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, scenes, labels, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
