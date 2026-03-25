from __future__ import annotations

import numpy as np
from s30bn.test_support import assert_array_shape

from s30bn.paper22_scaling_laws import (
    ScalingConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_scaling_data,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_scaling_law_backends_match_numpy() -> None:
    params = init_params()
    log_n, log_loss = synthetic_scaling_data()
    numpy_result = forward_numpy(params, log_n, log_loss)
    tinygrad_loss, tinygrad_predictions = forward_tinygrad(params_to_tinygrad(params), log_n, log_loss)
    torch_loss, torch_predictions = forward_torch(params_to_torch(params), log_n, log_loss)
    jax_loss, jax_predictions = forward_jax(params_to_jax(params), log_n, log_loss)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["predictions"], tinygrad_predictions.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["predictions"], torch_predictions.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["predictions"], np.asarray(jax_predictions), atol=1e-8)
    expected_shape = np.asarray(numpy_result["predictions"]).shape
    assert_array_shape(tinygrad_predictions.numpy(), expected_shape)
    assert_array_shape(torch_predictions.detach().numpy(), expected_shape)
    assert_array_shape(np.asarray(jax_predictions), expected_shape)


def test_scaling_law_one_step_improves_loss() -> None:
    config = ScalingConfig()
    params = init_params()
    log_n, log_loss = synthetic_scaling_data()

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, log_n, log_loss)
    train_step_tinygrad(tinygrad_params, log_n, log_loss, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, log_n, log_loss)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, log_n, log_loss)
    train_step_torch(torch_params, log_n, log_loss, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, log_n, log_loss)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, log_n, log_loss)
    _, jax_params = train_step_jax(jax_params, log_n, log_loss, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, log_n, log_loss)
    assert float(after_jax) <= float(before_jax)
