"""Tiny sparse MLP for paper 05."""

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
class PruningConfig:
    input_dim: int = 4
    hidden_dim: int = 4
    num_classes: int = 2
    learning_rate: float = 0.08


def synthetic_pruning_batch() -> Tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(
        [
            [1.0, 0.0, 0.8, 0.0],
            [0.0, 1.0, 0.0, 0.8],
            [0.9, 0.1, 0.7, 0.0],
            [0.1, 0.9, 0.0, 0.7],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return inputs, targets


def init_params(config: PruningConfig, seed: int = 5) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_hidden": rng.normal(scale=scale, size=(config.hidden_dim, config.input_dim)),
        "b_hidden": np.zeros((config.hidden_dim,), dtype=np.float64),
        "mask": np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "W_out": rng.normal(scale=scale, size=(config.num_classes, config.hidden_dim)),
        "b_out": np.zeros((config.num_classes,), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, inputs: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    sparse_w = params["W_hidden"] * params["mask"]
    hidden = np.tanh(inputs @ sparse_w.T + params["b_hidden"])
    logits = hidden @ params["W_out"].T + params["b_out"]
    probs = stable_softmax(logits, axis=1)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "logits": logits}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {
        "W_hidden": torch.tensor(params["W_hidden"], dtype=torch.float64, requires_grad=True),
        "b_hidden": torch.tensor(params["b_hidden"], dtype=torch.float64, requires_grad=True),
        "mask": torch.tensor(params["mask"], dtype=torch.float64, requires_grad=False),
        "W_out": torch.tensor(params["W_out"], dtype=torch.float64, requires_grad=True),
        "b_out": torch.tensor(params["b_out"], dtype=torch.float64, requires_grad=True),
    }


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {
        "W_hidden": Tensor(params["W_hidden"], requires_grad=True),
        "b_hidden": Tensor(params["b_hidden"], requires_grad=True),
        "mask": Tensor(params["mask"], requires_grad=False),
        "W_out": Tensor(params["W_out"], requires_grad=True),
        "b_out": Tensor(params["b_out"], requires_grad=True),
    }


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], inputs: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    x = Tensor(inputs.tolist(), dtype=params["W_hidden"].dtype)
    sparse_w = params["W_hidden"] * params["mask"]
    hidden = (x @ sparse_w.transpose() + params["b_hidden"]).tanh()
    logits = hidden @ params["W_out"].transpose() + params["b_out"]
    loss = logits.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits


def forward_torch(params: Dict[str, torch.Tensor], inputs: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(inputs, dtype=torch.float64)
    sparse_w = params["W_hidden"] * params["mask"]
    hidden = torch.tanh(x @ sparse_w.t() + params["b_hidden"])
    logits = hidden @ params["W_out"].t() + params["b_out"]
    loss = F.cross_entropy(logits, torch.tensor(targets))
    return loss, logits


def train_step_torch(params: Dict[str, torch.Tensor], inputs: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_torch(params, inputs, targets)
    loss.backward()
    with torch.no_grad():
        for name, tensor in params.items():
            if name == "mask":
                continue
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(params: Dict[str, Tensor], inputs: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_tinygrad(params, inputs, targets)
    loss.backward()
    for name, tensor in params.items():
        if name == "mask":
            continue
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(params: Dict[str, jnp.ndarray], inputs: np.ndarray, targets: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(inputs, dtype=jnp.float64)
    sparse_w = params["W_hidden"] * params["mask"]
    hidden = jnp.tanh(x @ sparse_w.T + params["b_hidden"])
    logits = hidden @ params["W_out"].T + params["b_out"]
    targets_j = jnp.asarray(targets)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    loss = -jnp.mean(log_probs[jnp.arange(targets_j.shape[0]), targets_j])
    return loss, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray], inputs: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, inputs, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = dict(params)
    for key in ("W_hidden", "b_hidden", "W_out", "b_out"):
        updated[key] = params[key] - learning_rate * grads[key]
    return float(loss.item()), updated
