from __future__ import annotations

import jax
import numpy as np
import torch
from s30bn.test_support import (
    assert_array_shape,
    assert_loss_trajectories_close,
    assert_parameter_dict_allclose,
    mapping_values_to_numpy,
)

from s30bn.paper20_neural_turing_machine import (
    NTMConfig,
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


def test_ntm_backends_match_numpy() -> None:
    config = NTMConfig()
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


def test_ntm_one_step_improves_loss() -> None:
    config = NTMConfig()
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


def test_ntm_one_step_losses_match_across_backends() -> None:
    config = NTMConfig()
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


def test_ntm_torch_and_jax_gradients_and_updated_params_match() -> None:
    config = NTMConfig()
    params = init_params(config)
    inputs, targets = build_dataset(config)

    torch_params = params_to_torch(params)
    torch_loss, _ = forward_torch(torch_params, inputs, targets, config)
    torch_loss.backward()
    torch_grads = mapping_values_to_numpy(torch_params, grad=True)

    jax_params = params_to_jax(params)
    jax_grads = jax.grad(lambda current: forward_jax(current, inputs, targets, config)[0])(jax_params)
    assert_parameter_dict_allclose(torch_grads, {"jax": jax_grads}, atol=1e-6)

    with torch.no_grad():
        for tensor in torch_params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    _, jax_updated = train_step_jax(jax_params, inputs, targets, config)

    assert_parameter_dict_allclose(mapping_values_to_numpy(torch_params), {"jax": jax_updated}, atol=1e-6)

    _, torch_logits = forward_torch(torch_params, inputs, targets, config)
    _, jax_logits = forward_jax(jax_updated, inputs, targets, config)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-6)
