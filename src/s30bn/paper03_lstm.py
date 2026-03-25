"""Tiny cross-backend LSTM for paper 03."""

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

from s30bn.common import ArrayDict, one_hot, seeded_rng

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LSTMConfig:
    vocab: str = "helo wrd"
    hidden_size: int = 5
    seq_len: int = 5
    learning_rate: float = 0.15

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def sample_text() -> str:
    return "hello hello world "


def encode_text(text: str, vocab: str) -> np.ndarray:
    mapping = {ch: idx for idx, ch in enumerate(vocab)}
    return np.asarray([mapping[ch] for ch in text], dtype=np.int64)


def build_dataset(config: LSTMConfig) -> Tuple[np.ndarray, np.ndarray]:
    encoded = encode_text(sample_text(), config.vocab)
    return encoded[: config.seq_len], encoded[1 : config.seq_len + 1]


def init_params(config: LSTMConfig, seed: int = 1) -> ArrayDict:
    rng = seeded_rng(seed)
    input_size = config.vocab_size + config.hidden_size
    scale = 0.2
    params = {}
    for name in ("Wf", "Wi", "Wo", "Wc"):
        params[name] = rng.normal(scale=scale, size=(config.hidden_size, input_size))
    for name in ("bf", "bi", "bo", "bc"):
        params[name] = np.zeros((config.hidden_size,), dtype=np.float64)
    params["Why"] = rng.normal(scale=scale, size=(config.vocab_size, config.hidden_size))
    params["by"] = np.zeros((config.vocab_size,), dtype=np.float64)
    return params


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def forward_numpy(
    params: ArrayDict,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    xs = one_hot(inputs, depth=params["by"].shape[0])
    h = np.zeros((params["bf"].shape[0],), dtype=np.float64)
    c = np.zeros_like(h)
    logits = []
    gates = {"f": [], "i": [], "o": []}
    for x_t in xs:
        z = np.concatenate([x_t, h])
        f_t = _sigmoid_np(params["Wf"] @ z + params["bf"])
        i_t = _sigmoid_np(params["Wi"] @ z + params["bi"])
        o_t = _sigmoid_np(params["Wo"] @ z + params["bo"])
        g_t = np.tanh(params["Wc"] @ z + params["bc"])
        c = f_t * c + i_t * g_t
        h = o_t * np.tanh(c)
        logits.append(params["Why"] @ h + params["by"])
        gates["f"].append(f_t)
        gates["i"].append(i_t)
        gates["o"].append(o_t)
    logits_arr = np.stack(logits)
    probs = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    loss = float(-np.mean(np.log(probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {
        "loss": loss,
        "logits": logits_arr,
        "forget_gate": np.stack(gates["f"]),
        "input_gate": np.stack(gates["i"]),
        "output_gate": np.stack(gates["o"]),
    }


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def forward_tinygrad(
    params: Dict[str, Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[Tensor, Tensor]:
    xs = one_hot(inputs, depth=int(params["by"].shape[0]))
    h = Tensor.zeros(int(params["bf"].shape[0]), dtype=params["bf"].dtype)
    c = Tensor.zeros(int(params["bf"].shape[0]), dtype=params["bf"].dtype)
    logits = []
    for x_t in xs:
        x_tensor = Tensor(x_t, dtype=params["Wf"].dtype)
        z = x_tensor.cat(h, dim=0)
        f_t = (params["Wf"] @ z + params["bf"]).sigmoid()
        i_t = (params["Wi"] @ z + params["bi"]).sigmoid()
        o_t = (params["Wo"] @ z + params["bo"]).sigmoid()
        g_t = (params["Wc"] @ z + params["bc"]).tanh()
        c = f_t * c + i_t * g_t
        h = o_t * c.tanh()
        logits.append(params["Why"] @ h + params["by"])
    logits_tensor = logits[0].stack(*logits[1:])
    loss = logits_tensor.sparse_categorical_crossentropy(Tensor(targets.tolist()))
    return loss, logits_tensor


def forward_torch(
    params: Dict[str, torch.Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = F.one_hot(torch.tensor(inputs), num_classes=params["by"].shape[0]).to(torch.float64)
    h = torch.zeros(params["bf"].shape[0], dtype=torch.float64)
    c = torch.zeros_like(h)
    logits = []
    for x_t in xs:
        z = torch.cat([x_t, h])
        f_t = torch.sigmoid(params["Wf"] @ z + params["bf"])
        i_t = torch.sigmoid(params["Wi"] @ z + params["bi"])
        o_t = torch.sigmoid(params["Wo"] @ z + params["bo"])
        g_t = torch.tanh(params["Wc"] @ z + params["bc"])
        c = f_t * c + i_t * g_t
        h = o_t * torch.tanh(c)
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


def train_step_tinygrad(
    params: Dict[str, Tensor],
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
) -> float:
    loss, _ = forward_tinygrad(params, inputs, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_jax(
    params: Dict[str, jnp.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xs = jax.nn.one_hot(jnp.asarray(inputs), params["by"].shape[0], dtype=jnp.float64)

    def scan_step(
        carry: Tuple[jnp.ndarray, jnp.ndarray],
        x_t: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        h, c = carry
        z = jnp.concatenate([x_t, h])
        f_t = jax.nn.sigmoid(params["Wf"] @ z + params["bf"])
        i_t = jax.nn.sigmoid(params["Wi"] @ z + params["bi"])
        o_t = jax.nn.sigmoid(params["Wo"] @ z + params["bo"])
        g_t = jnp.tanh(params["Wc"] @ z + params["bc"])
        new_c = f_t * c + i_t * g_t
        new_h = o_t * jnp.tanh(new_c)
        return (new_h, new_c), params["Why"] @ new_h + params["by"]

    init = (jnp.zeros_like(params["bf"]), jnp.zeros_like(params["bf"]))
    _, logits = jax.lax.scan(scan_step, init, xs)
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
