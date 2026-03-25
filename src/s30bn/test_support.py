"""Shared test helpers for parity, shapes, and simple invariants."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def assert_array_shape(array: np.ndarray, expected_shape: tuple[int, ...]) -> None:
    actual_shape = tuple(np.asarray(array).shape)
    assert actual_shape == expected_shape, f"expected shape {expected_shape}, got {actual_shape}"


def assert_allclose_named(
    reference: np.ndarray | float,
    comparisons: Mapping[str, np.ndarray | float],
    *,
    atol: float = 1e-8,
) -> None:
    reference_arr = np.asarray(reference)
    for name, value in comparisons.items():
        value_arr = np.asarray(value)
        assert np.allclose(reference_arr, value_arr, atol=atol), f"{name} mismatch"


def assert_probabilities_normalized(probs: np.ndarray, axis: int = -1, *, atol: float = 1e-8) -> None:
    totals = np.sum(np.asarray(probs), axis=axis)
    assert np.allclose(totals, 1.0, atol=atol), f"probabilities did not sum to 1 along axis {axis}"


def assert_loss_trajectories_close(
    reference: tuple[float, float],
    comparisons: Mapping[str, tuple[float, float]],
    *,
    atol: float = 1e-8,
) -> None:
    ref_before, ref_after = reference
    for name, (before, after) in comparisons.items():
        assert np.allclose(ref_before, before, atol=atol), f"{name} pre-step loss mismatch"
        assert np.allclose(ref_after, after, atol=atol), f"{name} post-step loss mismatch"


def assert_gradient_slice_close(
    reference: np.ndarray,
    comparisons: Mapping[str, np.ndarray],
    *,
    atol: float = 1e-7,
) -> None:
    reference_arr = np.asarray(reference, dtype=np.float64)
    for name, value in comparisons.items():
        value_arr = np.asarray(value, dtype=np.float64)
        assert value_arr.shape == reference_arr.shape, f"{name} gradient slice shape mismatch"
        assert np.allclose(reference_arr, value_arr, atol=atol), f"{name} gradient slice mismatch"


def permute_graph_batch(
    graphs: np.ndarray,
    adjacency: np.ndarray,
    permutation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    permuted_graphs = graphs[:, permutation, :]
    permuted_adjacency = adjacency[np.ix_(permutation, permutation)]
    return permuted_graphs, permuted_adjacency
