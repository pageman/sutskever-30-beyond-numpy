"""Tiny deterministic VAE for paper 17."""

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
class VAEConfig:
    input_dim: int = 4
    hidden_dim: int = 3
    latent_dim: int = 2
    learning_rate: float = 0.1
    epsilon: float = 0.25


def synthetic_binary_data() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def init_params(config: VAEConfig, seed: int = 17) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.22
    return {
        "W_enc": rng.normal(scale=scale, size=(config.hidden_dim, config.input_dim)),
        "b_enc": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_mu": rng.normal(scale=scale, size=(config.latent_dim, config.hidden_dim)),
        "b_mu": np.zeros((config.latent_dim,), dtype=np.float64),
        "W_logvar": rng.normal(scale=scale, size=(config.latent_dim, config.hidden_dim)),
        "b_logvar": np.zeros((config.latent_dim,), dtype=np.float64),
        "W_dec": rng.normal(scale=scale, size=(config.hidden_dim, config.latent_dim)),
        "b_dec": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.input_dim, config.hidden_dim)),
        "b_out": np.zeros((config.input_dim,), dtype=np.float64),
    }


def forward_numpy(
    params: ArrayDict,
    data: np.ndarray,
    config: VAEConfig,
) -> Dict[str, np.ndarray | float]:
    hidden = np.tanh(data @ params["W_enc"].T + params["b_enc"])
    mu = hidden @ params["W_mu"].T + params["b_mu"]
    logvar = hidden @ params["W_logvar"].T + params["b_logvar"]
    std = np.exp(0.5 * logvar)
    z = mu + config.epsilon * std
    dec_hidden = np.tanh(z @ params["W_dec"].T + params["b_dec"])
    logits = dec_hidden @ params["W_out"].T + params["b_out"]
    recon = 1.0 / (1.0 + np.exp(-logits))
    bce = -np.mean(np.sum(data * np.log(recon + 1e-12) + (1.0 - data) * np.log(1.0 - recon + 1e-12), axis=1))
    kl = 0.5 * np.mean(np.sum(np.exp(logvar) + mu**2 - 1.0 - logvar, axis=1))
    return {
        "loss": float(bce + kl),
        "logits": logits,
        "mu": mu,
        "logvar": logvar,
        "recon": recon,
        "bce": float(bce),
        "kl": float(kl),
    }


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    data: np.ndarray,
    config: VAEConfig,
) -> Tuple[Tensor, Tensor]:
    x = Tensor(data.tolist(), dtype=params["W_enc"].dtype)
    hidden = (x @ params["W_enc"].transpose() + params["b_enc"]).tanh()
    mu = hidden @ params["W_mu"].transpose() + params["b_mu"]
    logvar = hidden @ params["W_logvar"].transpose() + params["b_logvar"]
    std = (logvar * 0.5).exp()
    z = mu + std * config.epsilon
    dec_hidden = (z @ params["W_dec"].transpose() + params["b_dec"]).tanh()
    logits = dec_hidden @ params["W_out"].transpose() + params["b_out"]
    recon = logits.sigmoid()
    bce = -(x * recon.log() + (1 - x) * (1 - recon).log()).sum(axis=1).mean()
    kl = ((logvar.exp() + mu * mu - 1 - logvar).sum(axis=1) * 0.5).mean()
    return bce + kl, logits


def forward_torch(
    params: Dict[str, torch.Tensor],
    data: np.ndarray,
    config: VAEConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(data, dtype=torch.float64)
    hidden = torch.tanh(x @ params["W_enc"].t() + params["b_enc"])
    mu = hidden @ params["W_mu"].t() + params["b_mu"]
    logvar = hidden @ params["W_logvar"].t() + params["b_logvar"]
    std = torch.exp(0.5 * logvar)
    z = mu + config.epsilon * std
    dec_hidden = torch.tanh(z @ params["W_dec"].t() + params["b_dec"])
    logits = dec_hidden @ params["W_out"].t() + params["b_out"]
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
    kl = 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))
    return bce + kl, logits


def train_step_torch(
    params: Dict[str, torch.Tensor],
    data: np.ndarray,
    config: VAEConfig,
) -> float:
    loss, _ = forward_torch(params, data, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    data: np.ndarray,
    config: VAEConfig,
) -> float:
    loss, _ = forward_tinygrad(params, data, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    data: np.ndarray,
    config: VAEConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(data, dtype=jnp.float64)
    hidden = jnp.tanh(x @ params["W_enc"].T + params["b_enc"])
    mu = hidden @ params["W_mu"].T + params["b_mu"]
    logvar = hidden @ params["W_logvar"].T + params["b_logvar"]
    std = jnp.exp(0.5 * logvar)
    z = mu + config.epsilon * std
    dec_hidden = jnp.tanh(z @ params["W_dec"].T + params["b_dec"])
    logits = dec_hidden @ params["W_out"].T + params["b_out"]
    bce = jnp.mean(jnp.sum(jnp.maximum(logits, 0) - logits * x + jnp.log1p(jnp.exp(-jnp.abs(logits))), axis=1))
    kl = 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + mu**2 - 1.0 - logvar, axis=1))
    return bce + kl, logits


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    data: np.ndarray,
    config: VAEConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, data, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
