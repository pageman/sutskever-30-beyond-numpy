from __future__ import annotations

import numpy as np

from s30bn.paper11_dilated_convolutions import (
    DilatedConvConfig,
    forward_jax,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_context_dataset,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_dilated_convolution_backends_match() -> None:
    config = DilatedConvConfig()
    images, labels = synthetic_context_dataset(config)
    params = init_params(config)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), images, labels, config)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), images, labels, config)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), images, labels, config)
    assert np.allclose(tinygrad_loss.item(), float(jax_loss), atol=1e-8)
    assert np.allclose(tinygrad_logits.numpy(), np.asarray(jax_logits), atol=1e-8)
    assert np.allclose(float(torch_loss.detach()), float(jax_loss), atol=1e-8)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-8)


def test_dilated_convolution_one_step_improves_loss() -> None:
    config = DilatedConvConfig()
    images, labels = synthetic_context_dataset(config)
    params = init_params(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, images, labels, config)
    train_step_tinygrad(tinygrad_params, images, labels, config)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, images, labels, config)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, images, labels, config)
    train_step_torch(torch_params, images, labels, config)
    after_torch, _ = forward_torch(torch_params, images, labels, config)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, images, labels, config)
    _, jax_params = train_step_jax(jax_params, images, labels, config)
    after_jax, _ = forward_jax(jax_params, images, labels, config)
    assert float(after_jax) <= float(before_jax)
