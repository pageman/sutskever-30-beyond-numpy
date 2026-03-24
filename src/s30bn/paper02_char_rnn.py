"""Tiny cross-backend character RNN for paper 02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from s30bn.common import ArrayDict, one_hot, seeded_rng, stable_softmax

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class CharRNNConfig:
    vocab: str = "helo wrd"
    hidden_size: int = 6
    seq_len: int = 5
    learning_rate: float = 0.2

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def sample_text() -> str:
    return "hello world hello world "


def encode_text(text: str, vocab: str) -> np.ndarray:
    mapping = {ch: idx for idx, ch in enumerate(vocab)}
    return np.asarray([mapping[ch] for ch in text], dtype=np.int64)


def build_dataset(config: CharRNNConfig) -> Tuple[np.ndarray, np.ndarray]:
    encoded = encode_text(sample_text(), config.vocab)
    inputs = encoded[: config.seq_len]
    targets = encoded[1 : config.seq_len + 1]
    return inputs, targets


def init_params(config: CharRNNConfig, seed: int = 0) -> ArrayDict:
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
) -> Dict[str, np.ndarray | float]:
    xs = one_hot(inputs, depth=params["by"].shape[0])
    h = np.zeros((params["bh"].shape[0],), dtype=np.float64)
    logits = []
    hiddens = []
    for x_t in xs:
        h = np.tanh(params["Wxh"] @ x_t + params["Whh"] @ h + params["bh"])
        y_t = params["Why"] @ h + params["by"]
        logits.append(y_t)
        hiddens.append(h)
    logits_arr = np.stack(logits)
    probs = stable_softmax(logits_arr, axis=1)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "logits": logits_arr, "hidden": np.stack(hiddens), "probs": probs}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def forward_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = F.one_hot(torch.tensor(inputs), num_classes=params["by"].shape[0]).to(torch.float64)
    h = torch.zeros(params["bh"].shape[0], dtype=torch.float64)
    logits = []
    for x_t in xs:
        h = torch.tanh(params["Wxh"] @ x_t + params["Whh"] @ h + params["bh"])
        logits.append(params["Why"] @ h + params["by"])
    logits_tensor = torch.stack(logits)
    loss = F.cross_entropy(logits_tensor, torch.tensor(targets))
    return loss, logits_tensor


def train_step_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
) -> float:
    loss, _ = forward_torch(params, inputs, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xs = jax.nn.one_hot(jnp.asarray(inputs), params["by"].shape[0], dtype=jnp.float64)

    def scan_step(h: jnp.ndarray, x_t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        new_h = jnp.tanh(params["Wxh"] @ x_t + params["Whh"] @ h + params["bh"])
        return new_h, params["Why"] @ new_h + params["by"]

    _, logits = jax.lax.scan(scan_step, jnp.zeros_like(params["bh"]), xs)
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, inputs, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
