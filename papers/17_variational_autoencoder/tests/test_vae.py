from __future__ import annotations

import jax
import numpy as np
import torch

from s30bn.paper17_vae import (
    VAEConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_binary_data,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)
from s30bn.test_support import assert_allclose_named, assert_array_shape, assert_parameter_dict_allclose, mapping_values_to_numpy


def test_vae_backends_match_numpy() -> None:
    config = VAEConfig()
    params = init_params(config)
    data = synthetic_binary_data()
    numpy_result = forward_numpy(params, data, config)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), data, config)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), data, config)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), data, config)
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
    assert_array_shape(np.asarray(numpy_result["logits"]), data.shape)


def test_vae_one_step_improves_loss() -> None:
    config = VAEConfig()
    params = init_params(config)
    data = synthetic_binary_data()

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, data, config)
    train_step_tinygrad(tinygrad_params, data, config)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, data, config)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, data, config)
    train_step_torch(torch_params, data, config)
    after_torch, _ = forward_torch(torch_params, data, config)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, data, config)
    _, jax_params = train_step_jax(jax_params, data, config)
    after_jax, _ = forward_jax(jax_params, data, config)
    assert float(after_jax) <= float(before_jax)


def test_vae_kl_is_non_negative_and_reconstruction_shape_matches_input() -> None:
    config = VAEConfig()
    params = init_params(config)
    data = synthetic_binary_data()
    numpy_result = forward_numpy(params, data, config)

    assert float(numpy_result["kl"]) >= -1e-12
    assert float(numpy_result["bce"]) >= 0.0
    assert_array_shape(np.asarray(numpy_result["recon"]), data.shape)


def test_vae_torch_and_jax_gradients_and_updated_params_match() -> None:
    config = VAEConfig()
    params = init_params(config)
    data = synthetic_binary_data()

    torch_params = params_to_torch(params)
    torch_loss, _ = forward_torch(torch_params, data, config)
    torch_loss.backward()
    torch_grads = mapping_values_to_numpy(torch_params, grad=True)

    jax_params = params_to_jax(params)
    jax_grads = jax.grad(lambda current: forward_jax(current, data, config)[0])(jax_params)
    assert_parameter_dict_allclose(torch_grads, {"jax": jax_grads}, atol=1e-6)

    with torch.no_grad():
        for tensor in torch_params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    _, jax_updated = train_step_jax(jax_params, data, config)

    assert_parameter_dict_allclose(mapping_values_to_numpy(torch_params), {"jax": jax_updated}, atol=1e-6)

    _, torch_logits = forward_torch(torch_params, data, config)
    _, jax_logits = forward_jax(jax_updated, data, config)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-6)
