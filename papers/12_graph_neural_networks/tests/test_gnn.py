from __future__ import annotations

import numpy as np

from s30bn.paper12_gnn import (
    GNNConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_graph_dataset,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)
from s30bn.test_support import assert_allclose_named, assert_array_shape, permute_graph_batch


def test_gnn_backends_match() -> None:
    config = GNNConfig()
    graphs, adjacency, labels = synthetic_graph_dataset(config)
    params = init_params(config)
    numpy_result = forward_numpy(params, graphs, adjacency, labels)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), graphs, adjacency, labels)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), graphs, adjacency, labels)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), graphs, adjacency, labels)
    assert_allclose_named(
        numpy_result["loss"],
        {
            "tinygrad": tinygrad_loss.item(),
            "torch": float(torch_loss.detach()),
            "jax": float(jax_loss),
        },
    )
    assert_allclose_named(
        numpy_result["logits"],
        {
            "tinygrad": tinygrad_logits.numpy(),
            "torch": torch_logits.detach().numpy(),
            "jax": np.asarray(jax_logits),
        },
    )
    assert_array_shape(np.asarray(numpy_result["logits"]), (graphs.shape[0], config.num_classes))


def test_gnn_one_step_improves_loss() -> None:
    config = GNNConfig()
    graphs, adjacency, labels = synthetic_graph_dataset(config)
    params = init_params(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, graphs, adjacency, labels)
    train_step_tinygrad(tinygrad_params, graphs, adjacency, labels, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, graphs, adjacency, labels)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, graphs, adjacency, labels)
    train_step_torch(torch_params, graphs, adjacency, labels, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, graphs, adjacency, labels)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, graphs, adjacency, labels)
    _, jax_params = train_step_jax(jax_params, graphs, adjacency, labels, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, graphs, adjacency, labels)
    assert float(after_jax) <= float(before_jax)


def test_gnn_graph_level_logits_are_permutation_invariant() -> None:
    config = GNNConfig()
    graphs, adjacency, labels = synthetic_graph_dataset(config)
    params = init_params(config)
    permutation = np.asarray([2, 0, 1], dtype=np.int64)
    permuted_graphs, permuted_adjacency = permute_graph_batch(graphs, adjacency, permutation)

    base = forward_numpy(params, graphs, adjacency, labels)
    permuted = forward_numpy(params, permuted_graphs, permuted_adjacency, labels)

    assert_allclose_named(base["logits"], {"permuted": permuted["logits"]})
