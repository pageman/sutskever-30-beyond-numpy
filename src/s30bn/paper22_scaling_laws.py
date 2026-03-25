"""Tiny scaling-law regression for paper 22."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import os

import jax
import jax.numpy as jnp
import numpy as np
import torch

os.environ.setdefault("LLVM", "1")
from tinygrad import Tensor

from s30bn.common import ArrayDict, seeded_rng

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ScalingConfig:
    learning_rate: float = 0.005


def synthetic_scaling_data() -> Tuple[np.ndarray, np.ndarray]:
    n = np.asarray([1e3, 3e3, 1e4, 3e4], dtype=np.float64)
    loss = 1.5 * n ** (-0.3) + 0.2
    return np.log(n), np.log(loss)


def init_params(seed: int = 22) -> ArrayDict:
    rng = seeded_rng(seed)
    return {
        "slope": rng.normal(scale=0.1, size=(1,)).astype(np.float64),
        "intercept": rng.normal(scale=0.1, size=(1,)).astype(np.float64),
    }


def forward_numpy(params: ArrayDict, log_n: np.ndarray, log_loss: np.ndarray) -> Dict[str, np.ndarray | float]:
    pred = params["slope"][0] * log_n + params["intercept"][0]
    mse = np.mean((pred - log_loss) ** 2)
    return {"loss": float(mse), "predictions": pred}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], log_n: np.ndarray, log_loss: np.ndarray) -> Tuple[Tensor, Tensor]:
    x = Tensor(log_n.tolist(), dtype=params["slope"].dtype)
    y = Tensor(log_loss.tolist(), dtype=params["slope"].dtype)
    pred = params["slope"][0] * x + params["intercept"][0]
    loss = ((pred - y) * (pred - y)).mean()
    return loss, pred


def forward_torch(params: Dict[str, torch.Tensor], log_n: np.ndarray, log_loss: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(log_n, dtype=torch.float64)
    y = torch.tensor(log_loss, dtype=torch.float64)
    pred = params["slope"][0] * x + params["intercept"][0]
    loss = torch.mean((pred - y) ** 2)
    return loss, pred


def train_step_torch(params: Dict[str, torch.Tensor], log_n: np.ndarray, log_loss: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_torch(params, log_n, log_loss)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(params: Dict[str, Tensor], log_n: np.ndarray, log_loss: np.ndarray, learning_rate: float) -> float:
    loss, _ = forward_tinygrad(params, log_n, log_loss)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(params: Dict[str, jnp.ndarray], log_n: np.ndarray, log_loss: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(log_n, dtype=jnp.float64)
    y = jnp.asarray(log_loss, dtype=jnp.float64)
    pred = params["slope"][0] * x + params["intercept"][0]
    loss = jnp.mean((pred - y) ** 2)
    return loss, pred


def train_step_jax(
    params: Dict[str, jnp.ndarray], log_n: np.ndarray, log_loss: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, log_n, log_loss)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
