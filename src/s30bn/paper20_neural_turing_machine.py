"""Tiny content-addressed memory model for paper 20."""

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

from s30bn.common import ArrayDict, cross_entropy_from_probs, one_hot, seeded_rng, stable_softmax

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class NTMConfig:
    vocab: str = "abc "
    memory_slots: int = 3
    memory_width: int = 4
    seq_len: int = 3
    learning_rate: float = 0.08

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_dataset(config: NTMConfig) -> Tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray([0, 1, 2], dtype=np.int64)
    targets = np.asarray([1, 2, 0], dtype=np.int64)
    return inputs, targets


def init_params(config: NTMConfig, seed: int = 20) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.2
    return {
        "memory_init": rng.normal(scale=scale, size=(config.memory_slots, config.memory_width)),
        "W_key": rng.normal(scale=scale, size=(config.memory_width, config.vocab_size)),
        "b_key": np.zeros((config.memory_width,), dtype=np.float64),
        "W_add": rng.normal(scale=scale, size=(config.memory_width, config.vocab_size)),
        "b_add": np.zeros((config.memory_width,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.vocab_size, config.memory_width + config.vocab_size)),
        "b_out": np.zeros((config.vocab_size,), dtype=np.float64),
    }


def forward_numpy(
    params: ArrayDict,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> Dict[str, np.ndarray | float]:
    xs = one_hot(inputs, config.vocab_size)
    memory = params["memory_init"].copy()
    logits = []
    weights = []
    for x_t in xs:
        key = np.tanh(params["W_key"] @ x_t + params["b_key"])
        scores = memory @ key
        attention = stable_softmax(scores, axis=0)
        read = attention @ memory
        combined = np.concatenate([read, x_t])
        logits.append(params["W_out"] @ combined + params["b_out"])
        add = np.tanh(params["W_add"] @ x_t + params["b_add"])
        memory = memory + attention[:, None] * add[None, :]
        weights.append(attention)
    logits_arr = np.stack(logits)
    probs = stable_softmax(logits_arr, axis=1)
    loss = cross_entropy_from_probs(probs, targets)
    return {"loss": loss, "logits": logits_arr, "weights": np.stack(weights)}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def _softmax_tinygrad(x: Tensor) -> Tensor:
    ex = x.exp()
    return ex / ex.sum()


def forward_tinygrad(
    params: Dict[str, Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> Tuple[Tensor, Tensor]:
    xs = one_hot(inputs, config.vocab_size)
    memory = params["memory_init"]
    logits = []
    for x_t in xs:
        x_tensor = Tensor(x_t.tolist(), dtype=params["W_key"].dtype)
        key = (params["W_key"] @ x_tensor + params["b_key"]).tanh()
        scores = (memory * key.reshape(1, -1)).sum(axis=1)
        attention = _softmax_tinygrad(scores)
        read = (attention.reshape(-1, 1) * memory).sum(axis=0)
        combined = read.cat(x_tensor, dim=0)
        logits.append(params["W_out"] @ combined + params["b_out"])
        add = (params["W_add"] @ x_tensor + params["b_add"]).tanh()
        memory = memory + attention.reshape(-1, 1) * add.reshape(1, -1)
    logits_tensor = logits[0].stack(*logits[1:])
    loss = logits_tensor.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits_tensor


def forward_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = F.one_hot(torch.tensor(inputs), num_classes=config.vocab_size).to(torch.float64)
    memory = params["memory_init"]
    logits = []
    for x_t in xs:
        key = torch.tanh(params["W_key"] @ x_t + params["b_key"])
        scores = memory @ key
        attention = torch.softmax(scores, dim=0)
        read = attention @ memory
        combined = torch.cat([read, x_t])
        logits.append(params["W_out"] @ combined + params["b_out"])
        add = torch.tanh(params["W_add"] @ x_t + params["b_add"])
        memory = memory + attention.unsqueeze(1) * add.unsqueeze(0)
    logits_tensor = torch.stack(logits)
    loss = F.cross_entropy(logits_tensor, torch.tensor(targets))
    return loss, logits_tensor


def train_step_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> float:
    loss, _ = forward_torch(params, inputs, targets, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> float:
    loss, _ = forward_tinygrad(params, inputs, targets, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xs = jax.nn.one_hot(jnp.asarray(inputs), config.vocab_size, dtype=jnp.float64)

    def step(memory: jnp.ndarray, x_t: jnp.ndarray):
        key = jnp.tanh(params["W_key"] @ x_t + params["b_key"])
        scores = memory @ key
        attention = jax.nn.softmax(scores, axis=0)
        read = attention @ memory
        combined = jnp.concatenate([read, x_t])
        logits = params["W_out"] @ combined + params["b_out"]
        add = jnp.tanh(params["W_add"] @ x_t + params["b_add"])
        new_memory = memory + attention[:, None] * add[None, :]
        return new_memory, logits

    _, logits = jax.lax.scan(step, params["memory_init"], xs)
    labels_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: NTMConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, inputs, targets, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
