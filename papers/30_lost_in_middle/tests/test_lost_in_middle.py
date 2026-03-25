from __future__ import annotations

import numpy as np

from s30bn.paper30_lost_in_middle import (
    LostMiddleConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_middle_batch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_lost_in_middle_backends_match_numpy() -> None:
    config = LostMiddleConfig()
    params = init_params(config)
    chunks, queries, targets = synthetic_middle_batch()
    numpy_result = forward_numpy(params, chunks, queries, targets)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), chunks, queries, targets)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), chunks, queries, targets)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), chunks, queries, targets)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["logits"], tinygrad_logits.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], torch_logits.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], np.asarray(jax_logits), atol=1e-8)


def test_lost_in_middle_one_step_improves_loss() -> None:
    config = LostMiddleConfig()
    params = init_params(config)
    chunks, queries, targets = synthetic_middle_batch()

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, chunks, queries, targets)
    train_step_tinygrad(tinygrad_params, chunks, queries, targets, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, chunks, queries, targets)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, chunks, queries, targets)
    train_step_torch(torch_params, chunks, queries, targets, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, chunks, queries, targets)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, chunks, queries, targets)
    _, jax_params = train_step_jax(jax_params, chunks, queries, targets, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, chunks, queries, targets)
    assert float(after_jax) <= float(before_jax)
