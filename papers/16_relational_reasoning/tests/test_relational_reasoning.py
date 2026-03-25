from __future__ import annotations

import numpy as np
from s30bn.test_support import assert_array_shape

from s30bn.paper16_relational_reasoning import (
    RelationConfig,
    forward_jax,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_relation_dataset,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_relational_reasoning_backends_match() -> None:
    config = RelationConfig()
    scenes, labels = synthetic_relation_dataset(config)
    params = init_params(config)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), scenes, labels, config)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), scenes, labels, config)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), scenes, labels, config)
    assert np.allclose(tinygrad_loss.item(), float(jax_loss), atol=1e-8)
    assert np.allclose(tinygrad_logits.numpy(), np.asarray(jax_logits), atol=1e-8)
    assert np.allclose(float(torch_loss.detach()), float(jax_loss), atol=1e-8)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-8)
    expected_shape = (scenes.shape[0], config.num_classes)
    assert_array_shape(tinygrad_logits.numpy(), expected_shape)
    assert_array_shape(torch_logits.detach().numpy(), expected_shape)
    assert_array_shape(np.asarray(jax_logits), expected_shape)


def test_relational_reasoning_one_step_improves_loss() -> None:
    config = RelationConfig()
    scenes, labels = synthetic_relation_dataset(config)
    params = init_params(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, scenes, labels, config)
    train_step_tinygrad(tinygrad_params, scenes, labels, config)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, scenes, labels, config)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, scenes, labels, config)
    train_step_torch(torch_params, scenes, labels, config)
    after_torch, _ = forward_torch(torch_params, scenes, labels, config)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, scenes, labels, config)
    _, jax_params = train_step_jax(jax_params, scenes, labels, config)
    after_jax, _ = forward_jax(jax_params, scenes, labels, config)
    assert float(after_jax) <= float(before_jax)
