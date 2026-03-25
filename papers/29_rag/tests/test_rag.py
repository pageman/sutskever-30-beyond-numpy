from __future__ import annotations

import numpy as np
from s30bn.test_support import assert_array_shape

from s30bn.paper29_rag import (
    RAGConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_rag_batch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_rag_backends_match_numpy() -> None:
    config = RAGConfig()
    params = init_params(config)
    queries, passages, targets = synthetic_rag_batch()
    numpy_result = forward_numpy(params, queries, passages, targets)
    tinygrad_loss, tinygrad_probs = forward_tinygrad(params_to_tinygrad(params), queries, passages, targets)
    torch_loss, torch_probs = forward_torch(params_to_torch(params), queries, passages, targets)
    jax_loss, jax_probs = forward_jax(params_to_jax(params), queries, passages, targets)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["probs"], tinygrad_probs.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["probs"], torch_probs.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["probs"], np.asarray(jax_probs), atol=1e-8)
    expected_shape = np.asarray(numpy_result["probs"]).shape
    assert_array_shape(tinygrad_probs.numpy(), expected_shape)
    assert_array_shape(torch_probs.detach().numpy(), expected_shape)
    assert_array_shape(np.asarray(jax_probs), expected_shape)


def test_rag_one_step_improves_loss() -> None:
    config = RAGConfig()
    params = init_params(config)
    queries, passages, targets = synthetic_rag_batch()

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, queries, passages, targets)
    train_step_tinygrad(tinygrad_params, queries, passages, targets, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, queries, passages, targets)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, queries, passages, targets)
    train_step_torch(torch_params, queries, passages, targets, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, queries, passages, targets)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, queries, passages, targets)
    _, jax_params = train_step_jax(jax_params, queries, passages, targets, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, queries, passages, targets)
    assert float(after_jax) <= float(before_jax)
