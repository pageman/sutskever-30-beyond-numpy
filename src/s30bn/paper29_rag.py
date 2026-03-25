"""Tiny retrieval-augmented generation model for paper 29."""

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
class RAGConfig:
    query_dim: int = 3
    passage_dim: int = 3
    hidden_dim: int = 4
    vocab_size: int = 3
    num_docs: int = 3
    learning_rate: float = 0.06


def synthetic_rag_batch() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float64,
    )
    passages = np.asarray(
        [
            [[1.0, 0.0, 0.2], [0.0, 1.0, 0.2], [0.5, 0.5, 0.1]],
            [[0.0, 1.0, 0.2], [1.0, 0.0, 0.2], [0.5, 0.5, 0.1]],
            [[0.9, 0.1, 0.2], [0.1, 0.9, 0.2], [0.5, 0.5, 0.1]],
            [[0.1, 0.9, 0.2], [0.9, 0.1, 0.2], [0.5, 0.5, 0.1]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    return queries, passages, targets


def init_params(config: RAGConfig, seed: int = 29) -> ArrayDict:
    rng = seeded_rng(seed)
    scale = 0.18
    return {
        "W_q": rng.normal(scale=scale, size=(config.hidden_dim, config.query_dim)),
        "b_q": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_p": rng.normal(scale=scale, size=(config.hidden_dim, config.passage_dim)),
        "b_p": np.zeros((config.hidden_dim,), dtype=np.float64),
        "W_out": rng.normal(scale=scale, size=(config.vocab_size, 2 * config.hidden_dim)),
        "b_out": np.zeros((config.vocab_size,), dtype=np.float64),
    }


def forward_numpy(params: ArrayDict, queries: np.ndarray, passages: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray | float]:
    q = np.tanh(queries @ params["W_q"].T + params["b_q"])
    p = np.tanh(passages @ params["W_p"].T + params["b_p"])
    retrieval_scores = np.einsum("bh,bdh->bd", q, p)
    retrieval_weights = stable_softmax(retrieval_scores, axis=1)
    combined = np.concatenate([np.repeat(q[:, None, :], passages.shape[1], axis=1), p], axis=2)
    doc_logits = np.einsum("vh,bdh->bdv", params["W_out"], combined) + params["b_out"][None, None, :]
    doc_probs = stable_softmax(doc_logits, axis=2)
    mixture_probs = np.sum(retrieval_weights[:, :, None] * doc_probs, axis=1)
    loss = float(-np.mean(np.log(mixture_probs[np.arange(targets.shape[0]), targets] + 1e-12)))
    return {"loss": loss, "probs": mixture_probs}


def params_to_torch(params: ArrayDict) -> Dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float64, requires_grad=True) for key, value in params.items()}


def params_to_tinygrad(params: ArrayDict) -> Dict[str, Tensor]:
    return {key: Tensor(value, requires_grad=True) for key, value in params.items()}


def params_to_jax(params: ArrayDict) -> Dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value) for key, value in params.items()}


def forward_tinygrad(params: Dict[str, Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray) -> Tuple[Tensor, Tensor]:
    mixture_probs = []
    for query, docs in zip(queries, passages):
        q = (params["W_q"] @ Tensor(query.tolist(), dtype=params["W_q"].dtype) + params["b_q"]).tanh()
        doc_states = []
        doc_scores = []
        doc_probs = []
        for doc in docs:
            p = (params["W_p"] @ Tensor(doc.tolist(), dtype=params["W_p"].dtype) + params["b_p"]).tanh()
            doc_states.append(p)
            doc_scores.append((q * p).sum())
            combined = q.cat(p, dim=0)
            logits = params["W_out"] @ combined + params["b_out"]
            doc_probs.append(logits.softmax(axis=0))
        weights = doc_scores[0].stack(*doc_scores[1:]).softmax(axis=0)
        probs = (weights.reshape(-1, 1) * doc_probs[0].stack(*doc_probs[1:])).sum(axis=0)
        mixture_probs.append(probs)
    probs_tensor = mixture_probs[0].stack(*mixture_probs[1:])
    chosen = probs_tensor.gather(1, Tensor(targets.tolist()).reshape(-1, 1)).reshape(-1)
    loss = -(chosen.log()).mean()
    return loss, probs_tensor


def forward_torch(params: Dict[str, torch.Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.tanh(torch.tensor(queries, dtype=torch.float64) @ params["W_q"].t() + params["b_q"])
    p = torch.tanh(torch.tensor(passages, dtype=torch.float64) @ params["W_p"].t() + params["b_p"])
    retrieval_scores = torch.einsum("bh,bdh->bd", q, p)
    retrieval_weights = torch.softmax(retrieval_scores, dim=1)
    combined = torch.cat([q.unsqueeze(1).repeat(1, passages.shape[1], 1), p], dim=2)
    doc_logits = torch.einsum("vh,bdh->bdv", params["W_out"], combined) + params["b_out"].view(1, 1, -1)
    doc_probs = torch.softmax(doc_logits, dim=2)
    mixture_probs = torch.sum(retrieval_weights.unsqueeze(-1) * doc_probs, dim=1)
    loss = F.nll_loss(torch.log(mixture_probs + 1e-12), torch.tensor(targets))
    return loss, mixture_probs


def train_step_torch(
    params: Dict[str, torch.Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray, learning_rate: float
) -> float:
    loss, _ = forward_torch(params, queries, passages, targets)
    loss.backward()
    with torch.no_grad():
        for tensor in params.values():
            tensor -= learning_rate * tensor.grad
            tensor.grad.zero_()
    return float(loss.item())


def train_step_tinygrad(
    params: Dict[str, Tensor], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray, learning_rate: float
) -> float:
    loss, _ = forward_tinygrad(params, queries, passages, targets)
    loss.backward()
    for tensor in params.values():
        tensor.assign((tensor - learning_rate * tensor.grad).detach())
        tensor.grad = None
    return float(loss.item())


def forward_jax(
    params: Dict[str, jnp.ndarray], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q = jnp.tanh(jnp.asarray(queries, dtype=jnp.float64) @ params["W_q"].T + params["b_q"])
    p = jnp.tanh(jnp.asarray(passages, dtype=jnp.float64) @ params["W_p"].T + params["b_p"])
    retrieval_scores = jnp.einsum("bh,bdh->bd", q, p)
    retrieval_weights = jax.nn.softmax(retrieval_scores, axis=1)
    combined = jnp.concatenate([jnp.repeat(q[:, None, :], passages.shape[1], axis=1), p], axis=2)
    doc_logits = jnp.einsum("vh,bdh->bdv", params["W_out"], combined) + params["b_out"][None, None, :]
    doc_probs = jax.nn.softmax(doc_logits, axis=2)
    mixture_probs = jnp.sum(retrieval_weights[:, :, None] * doc_probs, axis=1)
    targets_j = jnp.asarray(targets)
    loss = -jnp.mean(jnp.log(mixture_probs[jnp.arange(targets_j.shape[0]), targets_j] + 1e-12))
    return loss, mixture_probs


def train_step_jax(
    params: Dict[str, jnp.ndarray], queries: np.ndarray, passages: np.ndarray, targets: np.ndarray, learning_rate: float
) -> Tuple[float, Dict[str, jnp.ndarray]]:
    def loss_fn(current: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        loss, _ = forward_jax(current, queries, passages, targets)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return float(loss.item()), updated
