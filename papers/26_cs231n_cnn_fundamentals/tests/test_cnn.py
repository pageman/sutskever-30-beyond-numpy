from __future__ import annotations

import numpy as np

from s30bn.paper26_cs231n import (
    CNNConfig,
    forward_jax,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_torch,
    synthetic_cifar_like,
    train_step_jax,
    train_step_torch,
)


def test_cnn_torch_and_jax_match() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = init_params(config)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), images, labels)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), images, labels)
    assert np.allclose(float(torch_loss.detach()), float(jax_loss), atol=1e-8)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-8)


def test_cnn_one_step_improves_loss() -> None:
    config = CNNConfig()
    images, labels = synthetic_cifar_like(config)
    params = init_params(config)

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
