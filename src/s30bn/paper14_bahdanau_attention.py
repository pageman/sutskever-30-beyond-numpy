"""Tiny Bahdanau-style additive attention model for paper 14."""

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
class BahdanauConfig:
    vocab: str = "abcd "
    hidden_size: int = 5
    seq_len: int = 4
    learning_rate: float = 0.12

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def sample_pair(config: BahdanauConfig) -> Tuple[np.ndarray, np.ndarray, int]:
    source = np.asarray([0, 1, 2, 3], dtype=np.int64)
    decoder_token = np.asarray([4], dtype=np.int64)
    target = 2
    return source, decoder_token, target


def init_params(config: BahdanauConfig, seed: int = 14) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_enc": rng.normal(scale=scale, size=(config.hidden_size, config.vocab_size)),
        "b_enc": np.zeros((config.hidden_size,), dtype=np.float64),
        "W_dec": rng.normal(scale=scale, size=(config.hidden_size, config.vocab_size)),
        "b_dec": np.zeros((config.hidden_size,), dtype=np.float64),
        "W_h": rng.normal(scale=scale, size=(config.hidden_size, config.hidden_size)),
        "W_s": rng.normal(scale=scale, size=(config.hidden_size, config.hidden_size)),
        "b_attn": np.zeros((config.hidden_size,), dtype=np.float64),
        "v_attn": rng.normal(scale=scale, size=(config.hidden_size,)),
        "W_out": rng.normal(scale=scale, size=(config.vocab_size, 2 * config.hidden_size)),
        "b_out": np.zeros((config.vocab_size,), dtype=np.float64),
    }


def forward_numpy(
    params: ArrayDict,
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> Dict[str, np.ndarray | float]:
    src = one_hot(source_tokens, config.vocab_size)
    dec = one_hot(decoder_token, config.vocab_size)[0]
    encoder_states = np.tanh(src @ params["W_enc"].T + params["b_enc"])
    decoder_state = np.tanh(params["W_dec"] @ dec + params["b_dec"])
    scores = []
    for h_i in encoder_states:
        score_hidden = np.tanh(params["W_h"] @ h_i + params["W_s"] @ decoder_state + params["b_attn"])
        scores.append(np.dot(params["v_attn"], score_hidden))
    scores_arr = np.asarray(scores, dtype=np.float64)
    attention = stable_softmax(scores_arr, axis=0)
    context = attention @ encoder_states
    combined = np.concatenate([context, decoder_state])
    logits = params["W_out"] @ combined + params["b_out"]
    probs = stable_softmax(logits, axis=0)
    loss = float(-np.log(probs[target] + 1e-12))
    return {"loss": loss, "logits": logits, "attention": attention}


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
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> Tuple[Tensor, Tensor, Tensor]:
    src = one_hot(source_tokens, config.vocab_size)
    dec = one_hot(decoder_token, config.vocab_size)[0]
    encoder_states = []
    for x_t in src:
        x_tensor = Tensor(x_t.tolist(), dtype=params["W_enc"].dtype)
        encoder_states.append((params["W_enc"] @ x_tensor + params["b_enc"]).tanh())
    dec_tensor = Tensor(dec.tolist(), dtype=params["W_dec"].dtype)
    decoder_state = (params["W_dec"] @ dec_tensor + params["b_dec"]).tanh()
    scores = []
    for h_i in encoder_states:
        score_hidden = (params["W_h"] @ h_i + params["W_s"] @ decoder_state + params["b_attn"]).tanh()
        scores.append((params["v_attn"] * score_hidden).sum())
    score_tensor = scores[0].stack(*scores[1:])
    attention = _softmax_tinygrad(score_tensor)
    context = (attention.reshape(-1, 1) * encoder_states[0].stack(*encoder_states[1:])).sum(axis=0)
    combined = context.cat(decoder_state, dim=0)
    logits = params["W_out"] @ combined + params["b_out"]
    loss = logits.reshape(1, -1).sparse_categorical_crossentropy(Tensor([target]))
    return loss, logits, attention


def forward_torch(
    params: Dict[str, torch.Tensor],
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = F.one_hot(torch.tensor(source_tokens), num_classes=config.vocab_size).to(torch.float64)
    dec = F.one_hot(torch.tensor(decoder_token), num_classes=config.vocab_size).to(torch.float64)[0]
    encoder_states = torch.tanh(src @ params["W_enc"].t() + params["b_enc"])
    decoder_state = torch.tanh(params["W_dec"] @ dec + params["b_dec"])
    scores = []
    for h_i in encoder_states:
        score_hidden = torch.tanh(params["W_h"] @ h_i + params["W_s"] @ decoder_state + params["b_attn"])
        scores.append(torch.dot(params["v_attn"], score_hidden))
    score_tensor = torch.stack(scores)
    attention = torch.softmax(score_tensor, dim=0)
    context = attention @ encoder_states
    combined = torch.cat([context, decoder_state])
    logits = params["W_out"] @ combined + params["b_out"]
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
    return loss, logits, attention


def train_step_torch(
    params: Dict[str, torch.Tensor],
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> float:
    loss, _, _ = forward_torch(params, source_tokens, decoder_token, target, config)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor],
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> float:
    loss, _, _ = forward_tinygrad(params, source_tokens, decoder_token, target, config)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - config.learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray],
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    src = jax.nn.one_hot(jnp.asarray(source_tokens), config.vocab_size, dtype=jnp.float64)
    dec = jax.nn.one_hot(jnp.asarray(decoder_token), config.vocab_size, dtype=jnp.float64)[0]
    encoder_states = jnp.tanh(src @ params["W_enc"].T + params["b_enc"])
    decoder_state = jnp.tanh(params["W_dec"] @ dec + params["b_dec"])

    def score_fn(h_i: jnp.ndarray) -> jnp.ndarray:
        score_hidden = jnp.tanh(params["W_h"] @ h_i + params["W_s"] @ decoder_state + params["b_attn"])
        return jnp.dot(params["v_attn"], score_hidden)

    scores = jax.vmap(score_fn)(encoder_states)
    attention = jax.nn.softmax(scores, axis=0)
    context = attention @ encoder_states
    combined = jnp.concatenate([context, decoder_state])
    logits = params["W_out"] @ combined + params["b_out"]
    log_probs = jax.nn.log_softmax(logits, axis=0)
    loss = -log_probs[target]
    return loss, logits, attention


def train_step_jax(
    params: Dict[str, jnp.ndarray],
    source_tokens: np.ndarray,
    decoder_token: np.ndarray,
    target: int,
    config: BahdanauConfig,
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _, _ = forward_jax(current, source_tokens, decoder_token, target, config)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - config.learning_rate * g, params, grads)
    return float(loss.item()), updated
