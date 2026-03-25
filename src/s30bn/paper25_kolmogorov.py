"""Tiny compressibility classifier for paper 25."""

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
class KolmogorovConfig:
    seq_len: int = 6
    hidden_dim: int = 4
    num_classes: int = 2
    learning_rate: float = 0.08


def synthetic_kolmogorov_batch() -> Tuple[np.ndarray, np.ndarray]:
    sequences = np.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 1],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    return sequences, targets


def init_params(config: KolmogorovConfig, seed: int = 25) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_hidden": rng.normal(scale=scale, size=(config.hidden_dim, config.seq_len)),
        "b_hidden": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.num_classes, config.hidden_dim)),
        "b_out": np.zeros((config.num_classes,), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, sequences: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    hidden = np.tanh(sequences @ params["W_hidden"].T + params["b_hidden"])
    logits = hidden @ params["W_out"].T + params["b_out"]
    probs = stable_softmax(logits, axis=1)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "logits": logits}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], sequences: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    x = Tensor(sequences.tolist(), dtype=params["W_hidden"].dtype)
    hidden = (x @ params["W_hidden"].transpose() + params["b_hidden"]).tanh()
    logits = hidden @ params["W_out"].transpose() + params["b_out"]
    loss = logits.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits


def forward_torch(params: Dict[str, torch.Tensor], sequences: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(sequences, dtype=torch.float64)
    hidden = torch.tanh(x @ params["W_hidden"].t() + params["b_hidden"])
    logits = hidden @ params["W_out"].t() + params["b_out"]
    loss = F.cross_entropy(logits, torch.tensor(targets))
    return loss, logits


def train_step_torch(params: Dict[str, torch.Tensor], sequences: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_torch(params, sequences, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(params: Dict[str, Tensor], sequences: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_tinygrad(params, sequences, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(params: Dict[str, jnp.ndarray], sequences: np.ndarray, targets: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(sequences, dtype=jnp.float64)
    hidden = jnp.tanh(x @ params["W_hidden"].T + params["b_hidden"])
    logits = hidden @ params["W_out"].T + params["b_out"]
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray], sequences: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, sequences, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
