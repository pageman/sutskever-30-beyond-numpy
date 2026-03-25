"""Tiny multi-token prediction model for paper 27."""

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

from s30bn.common import ArrayDict, cross_entropy_from_probs, seeded_rng, stable_softmax

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class MultiTokenConfig:
    input_dim: int = 3
    hidden_dim: int = 5
    vocab_size: int = 4
    num_tokens: int = 2
    learning_rate: float = 0.08


def synthetic_mtp_batch() -> Tuple[np.ndarray, np.ndarray]:
    contexts = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    return contexts, targets


def init_params(config: MultiTokenConfig, seed: int = 27) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_hidden": rng.normal(scale=scale, size=(config.hidden_dim, config.input_dim)),
        "b_hidden": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.num_tokens, config.vocab_size, config.hidden_dim)),
        "b_out": np.zeros((config.num_tokens, config.vocab_size), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, contexts: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    hidden = np.tanh(contexts @ params["W_hidden"].T + params["b_hidden"])
    logits = np.einsum("tvh,bh->btv", params["W_out"], hidden) + params["b_out"][None, :, :]
    probs = stable_softmax(logits, axis=2)
    losses = []
    for token_idx in range(targets.shape[1]):
        losses.append(cross_entropy_from_probs(probs[:, token_idx, :], targets[:, token_idx]))
    return {"loss": float(np.mean(losses)), "logits": logits}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], contexts: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    batch_logits = []
    for context in contexts:
        x = Tensor(context.tolist(), dtype=params["W_hidden"].dtype)
        hidden = (params["W_hidden"] @ x + params["b_hidden"]).tanh()
        token_logits = []
        for token_idx in range(targets.shape[1]):
            token_logits.append(params["W_out"][token_idx] @ hidden + params["b_out"][token_idx])
        batch_logits.append(token_logits[0].stack(*token_logits[1:]))
    logits = batch_logits[0].stack(*batch_logits[1:])
    vocab_size = int(params["b_out"].shape[1])
    flat_logits = logits.reshape(-1, vocab_size)
    loss = flat_logits.sparse_categorical_crossentropy(Tensor(targets.reshape(-1).tolist()))
    return loss, logits


def forward_torch(params: Dict[str, torch.Tensor], contexts: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(contexts, dtype=torch.float64)
    hidden = torch.tanh(x @ params["W_hidden"].t() + params["b_hidden"])
    logits = torch.einsum("tvh,bh->btv", params["W_out"], hidden) + params["b_out"].unsqueeze(0)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(targets).reshape(-1))
    return loss, logits


def train_step_torch(params: Dict[str, torch.Tensor], contexts: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_torch(params, contexts, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(params: Dict[str, Tensor], contexts: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_tinygrad(params, contexts, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(params: Dict[str, jnp.ndarray], contexts: np.ndarray, targets: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(contexts, dtype=jnp.float64)
    hidden = jnp.tanh(x @ params["W_hidden"].T + params["b_hidden"])
    logits = jnp.einsum("tvh,bh->btv", params["W_out"], hidden) + params["b_out"][None, :, :]
    targets_j = jnp.asarray(targets).reshape(-1)
    log_probs = jax.nn.log_softmax(logits.reshape(-1, logits.shape[-1]), axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray], contexts: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, contexts, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
