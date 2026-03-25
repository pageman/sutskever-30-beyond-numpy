from __future__ import annotations

import numpy as np

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


def test_vae_backends_match_numpy() -> None:
    config = VAEConfig()
    params = init_params(config)
    data = synthetic_binary_data()
    numpy_result = forward_numpy(params, data, config)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), data, config)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), data, config)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), data, config)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["logits"], tinygrad_logits.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], torch_logits.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], np.asarray(jax_logits), atol=1e-8)


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
