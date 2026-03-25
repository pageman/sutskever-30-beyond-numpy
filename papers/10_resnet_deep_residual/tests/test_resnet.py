from __future__ import annotations

import numpy as np

from s30bn.paper10_resnet import (
    ResNetConfig,
    forward_jax,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_residual_dataset,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_resnet_backends_match() -> None:
    config = ResNetConfig()
    images, labels = synthetic_residual_dataset(config)
    params = init_params(config)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), images, labels)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), images, labels)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), images, labels)
    assert np.allclose(tinygrad_loss.item(), float(jax_loss), atol=1e-8)
    assert np.allclose(tinygrad_logits.numpy(), np.asarray(jax_logits), atol=1e-8)
    assert np.allclose(float(torch_loss.detach()), float(jax_loss), atol=1e-8)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-8)


def test_resnet_one_step_improves_loss() -> None:
    config = ResNetConfig()
    images, labels = synthetic_residual_dataset(config)
    params = init_params(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, images, labels)
    train_step_tinygrad(tinygrad_params, images, labels, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, images, labels)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, images, labels)
    train_step_torch(torch_params, images, labels, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, images, labels)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, images, labels)
    _, jax_params = train_step_jax(jax_params, images, labels, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, images, labels)
    assert float(after_jax) <= float(before_jax)
