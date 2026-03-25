"""Tiny relational RNN for paper 18."""

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
class RelationalRNNConfig:
    vocab: str = "helo wrd"
    hidden_size: int = 4
    num_slots: int = 2
    seq_len: int = 4
    learning_rate: float = 0.1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def sample_text() -> str:
    return "hello world"


def encode_text(text: str, vocab: str) -> np.ndarray:
    mapping = {ch: idx for idx, ch in enumerate(vocab)}
    return np.asarray([mapping[ch] for ch in text], dtype=np.int64)


def build_dataset(config: RelationalRNNConfig) -> Tuple[np.ndarray, np.ndarray]:
    encoded = encode_text(sample_text(), config.vocab)
    return encoded[: config.seq_len], encoded[1 : config.seq_len + 1]


def init_params(config: RelationalRNNConfig, seed: int = 18) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    pair_dim = 2 * config.hidden_size
    return {
        "memory": rng.normal(scale=scale, size=(config.num_slots, config.hidden_size)),
        "W_rel": rng.normal(scale=scale, size=(config.hidden_size, pair_dim)),
        "b_rel": np.zeros((config.hidden_size,), dtype=np.float64),
        "W_mem": rng.normal(scale=scale, size=(config.hidden_size, config.hidden_size)),
        "W_x": rng.normal(scale=scale, size=(config.hidden_size, config.vocab_size)),
        "b_h": np.zeros((config.hidden_size,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.vocab_size, config.hidden_size)),
        "b_out": np.zeros((config.vocab_size,), dtype=np.float64),
    }


def forward_numpy(
    params: ArrayDict,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RelationalRNNConfig,
) -> Dict[str, np.ndarray | float]:
    xs = one_hot(inputs, config.vocab_size)
    memory = params["memory"].copy()
    logits = []
    for x_t in xs:
        relations = []
        for i in range(config.num_slots):
            for j in range(config.num_slots):
                pair = np.concatenate([memory[i], memory[j]])
                relations.append(np.tanh(params["W_rel"] @ pair + params["b_rel"]))
        relation_summary = np.mean(relations, axis=0)
        pooled_memory = np.mean(memory, axis=0)
        new_slot = np.tanh(params["W_mem"] @ pooled_memory + params["W_x"] @ x_t + relation_summary + params["b_h"])
        memory = np.stack([new_slot, 0.5 * (memory[0] + new_slot)])
        pooled_memory = np.mean(memory, axis=0)
        logits.append(params["W_out"] @ pooled_memory + params["b_out"])
    logits_arr = np.stack(logits)
    probs = stable_softmax(logits_arr, axis=1)
    loss = cross_entropy_from_probs(probs, targets)
    return {"loss": loss, "logits": logits_arr}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RelationalRNNConfig,
) -> Tuple[Tensor, Tensor]:
    xs = one_hot(inputs, config.vocab_size)
    memory = params["memory"]
    logits = []
    for x_t in xs:
        relations = []
        for i in range(config.num_slots):
            for j in range(config.num_slots):
                pair = memory[i].cat(memory[j], dim=0)
                relations.append((params["W_rel"] @ pair + params["b_rel"]).tanh())
        relation_summary = relations[0].stack(*relations[1:]).mean(axis=0)
        pooled_memory = memory.mean(axis=0)
        x_tensor = Tensor(x_t.tolist(), dtype=params["W_x"].dtype)
        new_slot = (params["W_mem"] @ pooled_memory + params["W_x"] @ x_tensor + relation_summary + params["b_h"]).tanh()
        memory = new_slot.stack((memory[0] + new_slot) * 0.5)
        pooled_memory = memory.mean(axis=0)
        logits.append(params["W_out"] @ pooled_memory + params["b_out"])
    logits_tensor = logits[0].stack(*logits[1:])
    loss = logits_tensor.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits_tensor


def forward_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RelationalRNNConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = F.one_hot(torch.tensor(inputs), num_classes=config.vocab_size).to(torch.float64)
    memory = params["memory"]
    logits = []
    for x_t in xs:
        relations = []
        for i in range(config.num_slots):
            for j in range(config.num_slots):
                pair = torch.cat([memory[i], memory[j]])
                relations.append(torch.tanh(params["W_rel"] @ pair + params["b_rel"]))
        relation_summary = torch.stack(relations).mean(dim=0)
        pooled_memory = memory.mean(dim=0)
        new_slot = torch.tanh(params["W_mem"] @ pooled_memory + params["W_x"] @ x_t + relation_summary + params["b_h"])
        memory = torch.stack([new_slot, 0.5 * (memory[0] + new_slot)])
        pooled_memory = memory.mean(dim=0)
        logits.append(params["W_out"] @ pooled_memory + params["b_out"])
    logits_tensor = torch.stack(logits)
    loss = F.cross_entropy(logits_tensor, torch.tensor(targets))
    return loss, logits_tensor


def train_step_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RelationalRNNConfig,
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
    config: RelationalRNNConfig,
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
    config: RelationalRNNConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xs = jax.nn.one_hot(jnp.asarray(inputs), config.vocab_size, dtype=jnp.float64)

    def step(memory: jnp.ndarray, x_t: jnp.ndarray):
        relations = []
        for i in range(config.num_slots):
            for j in range(config.num_slots):
                pair = jnp.concatenate([memory[i], memory[j]])
                relations.append(jnp.tanh(params["W_rel"] @ pair + params["b_rel"]))
        relation_summary = jnp.mean(jnp.stack(relations), axis=0)
        pooled_memory = jnp.mean(memory, axis=0)
        new_slot = jnp.tanh(params["W_mem"] @ pooled_memory + params["W_x"] @ x_t + relation_summary + params["b_h"])
        new_memory = jnp.stack([new_slot, 0.5 * (memory[0] + new_slot)])
        pooled_new = jnp.mean(new_memory, axis=0)
        logits = params["W_out"] @ pooled_new + params["b_out"]
        return new_memory, logits

    _, logits = jax.lax.scan(step, params["memory"], xs)
    labels_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(labels_j.shape[0]), labels_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RelationalRNNConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, inputs, targets, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
