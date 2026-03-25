from __future__ import annotations

import numpy as np

from s30bn.paper28_dense_passage_retrieval import (
    DPRConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_dpr_batch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_dpr_backends_match_numpy() -> None:
    config = DPRConfig()
    params = init_params(config)
    queries, passages, targets = synthetic_dpr_batch()
    numpy_result = forward_numpy(params, queries, passages, targets)
    tinygrad_loss, tinygrad_scores = forward_tinygrad(params_to_tinygrad(params), queries, passages, targets)
    torch_loss, torch_scores = forward_torch(params_to_torch(params), queries, passages, targets)
    jax_loss, jax_scores = forward_jax(params_to_jax(params), queries, passages, targets)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["scores"], tinygrad_scores.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["scores"], torch_scores.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["scores"], np.asarray(jax_scores), atol=1e-8)


def test_dpr_one_step_improves_loss() -> None:
    config = DPRConfig()
    params = init_params(config)
    queries, passages, targets = synthetic_dpr_batch()

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
