"""Small shared helpers used by multiple papers."""

from __future__ import annotations

from typing import Dict

import numpy as np


ArrayDict = Dict[str, np.ndarray]


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    eye = np.eye(depth, dtype=np.float64)
    return eye[np.asarray(indices, dtype=np.int64)]


def cross_entropy_from_probs(probs: np.ndarray, targets: np.ndarray) -> float:
    target_probs = probs[np.arange(targets.shape[0]), targets]
    return float(-np.mean(np.log(target_probs + 1e-12)))


def seeded_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

