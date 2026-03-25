from __future__ import annotations

import numpy as np
from s30bn.test_support import assert_array_shape, assert_loss_trajectories_close

from s30bn.paper04_rnn_regularization import (
    RNNRegularizationConfig,
    build_dataset,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_rnn_regularization_backends_match_numpy() -> None:
    config = RNNRegularizationConfig()
    params = init_params(config)
    inputs, targets = build_dataset(config)
    numpy_result = forward_numpy(params, inputs, targets, config)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), inputs, targets, config)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), inputs, targets, config)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), inputs, targets, config)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["logits"], tinygrad_logits.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], torch_logits.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], np.asarray(jax_logits), atol=1e-8)
    expected_shape = np.asarray(numpy_result["logits"]).shape
    assert_array_shape(tinygrad_logits.numpy(), expected_shape)
    assert_array_shape(torch_logits.detach().numpy(), expected_shape)
    assert_array_shape(np.asarray(jax_logits), expected_shape)


def test_rnn_regularization_one_step_improves_loss() -> None:
    config = RNNRegularizationConfig()
    params = init_params(config)
    inputs, targets = build_dataset(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, inputs, targets, config)
    train_step_tinygrad(tinygrad_params, inputs, targets, config)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, inputs, targets, config)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, inputs, targets, config)
    train_step_torch(torch_params, inputs, targets, config)
    after_torch, _ = forward_torch(torch_params, inputs, targets, config)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, inputs, targets, config)
    _, jax_params = train_step_jax(jax_params, inputs, targets, config)
    after_jax, _ = forward_jax(jax_params, inputs, targets, config)
    assert float(after_jax) <= float(before_jax)


def test_rnn_regularization_one_step_losses_match_across_backends() -> None:
    config = RNNRegularizationConfig()
    params = init_params(config)
    inputs, targets = build_dataset(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, inputs, targets, config)
    train_step_tinygrad(tinygrad_params, inputs, targets, config)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, inputs, targets, config)

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, inputs, targets, config)
    train_step_torch(torch_params, inputs, targets, config)
    after_torch, _ = forward_torch(torch_params, inputs, targets, config)

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, inputs, targets, config)
    _, jax_params = train_step_jax(jax_params, inputs, targets, config)
    after_jax, _ = forward_jax(jax_params, inputs, targets, config)

    assert_loss_trajectories_close(
        (float(before_torch.detach()), float(after_torch.detach())),
        {
            "tinygrad": (before_tinygrad.item(), after_tinygrad.item()),
            "jax": (float(before_jax), float(after_jax)),
        },
    )
