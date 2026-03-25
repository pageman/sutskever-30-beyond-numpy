"""Tiny pointer network for paper 06."""

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
class PointerConfig:
    seq_len: int = 4
    item_dim: int = 2
    hidden_dim: int = 4
    learning_rate: float = 0.08


def synthetic_pointer_batch() -> Tuple[np.ndarray, np.ndarray]:
    seqs = np.asarray(
        [
            [[0.9, 0.1], [0.3, 0.2], [1.0, 0.0], [0.2, 0.8]],
            [[0.2, 0.9], [0.1, 1.0], [0.8, 0.2], [0.3, 0.7]],
            [[0.7, 0.2], [0.9, 0.1], [0.4, 0.6], [0.3, 0.8]],
            [[0.1, 0.7], [0.2, 0.8], [0.9, 0.0], [0.8, 0.1]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([2, 1, 1, 2], dtype=np.int64)
    return seqs, targets


def init_params(config: PointerConfig, seed: int = 6) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_enc": rng.normal(scale=scale, size=(config.hidden_dim, config.item_dim)),
        "b_enc": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_query": rng.normal(scale=scale, size=(config.hidden_dim, config.item_dim)),
        "b_query": np.zeros((config.hidden_dim,), dtype=np.float64),
        "v": rng.normal(scale=scale, size=(config.hidden_dim,)),
    }


def forward_numpy(params: ArrayDict, seqs: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    encoder = np.tanh(seqs @ params["W_enc"].T + params["b_enc"])
    query = np.tanh(np.mean(seqs, axis=1) @ params["W_query"].T + params["b_query"])
    scores = np.einsum("bsh,h->bs", np.tanh(encoder + query[:, None, :]), params["v"])
    probs = stable_softmax(scores, axis=1)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "logits": scores}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], seqs: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    scores = []
    for seq in seqs:
        x = Tensor(seq.tolist(), dtype=params["W_enc"].dtype)
        encoder = (x @ params["W_enc"].transpose() + params["b_enc"]).tanh()
        mean_input = x.mean(axis=0)
        query = (params["W_query"] @ mean_input + params["b_query"]).tanh()
        score = ((encoder + query.reshape(1, -1)).tanh() * params["v"].reshape(1, -1)).sum(axis=1)
        scores.append(score)
    logits = scores[0].stack(*scores[1:])
    loss = logits.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits


def forward_torch(params: Dict[str, torch.Tensor], seqs: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(seqs, dtype=torch.float64)
    encoder = torch.tanh(x @ params["W_enc"].t() + params["b_enc"])
    query = torch.tanh(x.mean(dim=1) @ params["W_query"].t() + params["b_query"])
    logits = torch.sum(torch.tanh(encoder + query.unsqueeze(1)) * params["v"], dim=2)
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
    encoder = jnp.tanh(x @ params["W_enc"].T + params["b_enc"])
    query = jnp.tanh(jnp.mean(x, axis=1) @ params["W_query"].T + params["b_query"])
    logits = jnp.sum(jnp.tanh(encoder + query[:, None, :]) * params["v"], axis=2)
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
