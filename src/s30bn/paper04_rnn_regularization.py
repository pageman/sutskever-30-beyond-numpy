"""Deterministic dropout-regularized RNN for paper 04."""

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
class RNNRegularizationConfig:
    vocab: str = "helo wrd"
    hidden_size: int = 6
    seq_len: int = 6
    learning_rate: float = 0.2
    input_keep_prob: float = 0.75
    hidden_keep_prob: float = 0.8

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def sample_text() -> str:
    return "hello world hello "


def encode_text(text: str, vocab: str) -> np.ndarray:
    mapping = {ch: idx for idx, ch in enumerate(vocab)}
    return np.asarray([mapping[ch] for ch in text], dtype=np.int64)


def build_dataset(config: RNNRegularizationConfig) -> Tuple[np.ndarray, np.ndarray]:
    encoded = encode_text(sample_text(), config.vocab)
    return encoded[: config.seq_len], encoded[1 : config.seq_len + 1]


def regularization_masks(config: RNNRegularizationConfig) -> Tuple[np.ndarray, np.ndarray]:
    input_mask = np.full((config.vocab_size,), config.input_keep_prob, dtype=np.float64)
    hidden_mask = np.full((config.hidden_size,), config.hidden_keep_prob, dtype=np.float64)
    return input_mask, hidden_mask


def init_params(config: RNNRegularizationConfig, seed: int = 4) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.2
    return {
        "Wxh": rng.normal(scale=scale, size=(config.hidden_size, config.vocab_size)),
        "Whh": rng.normal(scale=scale, size=(config.hidden_size, config.hidden_size)),
        "Why": rng.normal(scale=scale, size=(config.vocab_size, config.hidden_size)),
        "bh": np.zeros((config.hidden_size,), dtype=np.float64),
        "by": np.zeros((config.vocab_size,), dtype=np.float64),
    }


def forward_numpy(
    params: ArrayDict,
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RNNRegularizationConfig,
) -> Dict[str, np.ndarray | float]:
    xs = one_hot(inputs, config.vocab_size)
    input_mask, hidden_mask = regularization_masks(config)
    h = np.zeros((config.hidden_size,), dtype=np.float64)
    logits = []
    hidden_states = []
    for x_t in xs:
        masked_x = x_t * input_mask
        masked_h = h * hidden_mask
        h = np.tanh(params["Wxh"] @ masked_x + params["Whh"] @ masked_h + params["bh"])
        logits.append(params["Why"] @ h + params["by"])
        hidden_states.append(h.copy())
    logits_arr = np.stack(logits)
    probs = stable_softmax(logits_arr, axis=1)
    loss = cross_entropy_from_probs(probs, targets)
    return {
        "loss": loss,
        "logits": logits_arr,
        "hidden_states": np.stack(hidden_states),
        "input_mask": input_mask,
        "hidden_mask": hidden_mask,
    }


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
    config: RNNRegularizationConfig,
) -> Tuple[Tensor, Tensor]:
    xs = one_hot(inputs, config.vocab_size)
    input_mask, hidden_mask = regularization_masks(config)
    input_mask_t = Tensor(input_mask.tolist(), dtype=params["bh"].dtype)
    hidden_mask_t = Tensor(hidden_mask.tolist(), dtype=params["bh"].dtype)
    h = Tensor.zeros(config.hidden_size, dtype=params["bh"].dtype)
    logits = []
    for x_t in xs:
        x_tensor = Tensor(x_t.tolist(), dtype=params["bh"].dtype)
        masked_x = x_tensor * input_mask_t
        masked_h = h * hidden_mask_t
        h = (params["Wxh"] @ masked_x + params["Whh"] @ masked_h + params["bh"]).tanh()
        logits.append(params["Why"] @ h + params["by"])
    logits_tensor = logits[0].stack(*logits[1:])
    loss = logits_tensor.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits_tensor


def forward_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RNNRegularizationConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = F.one_hot(torch.tensor(inputs), num_classes=config.vocab_size).to(torch.float64)
    input_mask, hidden_mask = regularization_masks(config)
    input_mask_t = torch.tensor(input_mask, dtype=torch.float64)
    hidden_mask_t = torch.tensor(hidden_mask, dtype=torch.float64)
    h = torch.zeros(config.hidden_size, dtype=torch.float64)
    logits = []
    for x_t in xs:
        masked_x = x_t * input_mask_t
        masked_h = h * hidden_mask_t
        h = torch.tanh(params["Wxh"] @ masked_x + params["Whh"] @ masked_h + params["bh"])
        logits.append(params["Why"] @ h + params["by"])
    logits_tensor = torch.stack(logits)
    loss = F.cross_entropy(logits_tensor, torch.tensor(targets))
    return loss, logits_tensor


def train_step_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RNNRegularizationConfig,
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
    config: RNNRegularizationConfig,
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
    config: RNNRegularizationConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xs = jax.nn.one_hot(jnp.asarray(inputs), config.vocab_size, dtype=jnp.float64)
    input_mask, hidden_mask = regularization_masks(config)
    input_mask_j = jnp.asarray(input_mask)
    hidden_mask_j = jnp.asarray(hidden_mask)

    def scan_step(h: jnp.ndarray, x_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        masked_x = x_t * input_mask_j
        masked_h = h * hidden_mask_j
        new_h = jnp.tanh(params["Wxh"] @ masked_x + params["Whh"] @ masked_h + params["bh"])
        return new_h, params["Why"] @ new_h + params["by"]

    init = jnp.zeros((config.hidden_size,), dtype=jnp.float64)
    _, logits = jax.lax.scan(scan_step, init, xs)
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    config: RNNRegularizationConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, inputs, targets, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
