from __future__ import annotations

import numpy as np
from s30bn.test_support import assert_array_shape

from s30bn.paper06_pointer_networks import (
    PointerConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_pointer_batch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_pointer_network_backends_match_numpy() -> None:
    config = PointerConfig()
    params = init_params(config)
    seqs, targets = synthetic_pointer_batch()
    numpy_result = forward_numpy(params, seqs, targets)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), seqs, targets)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), seqs, targets)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), seqs, targets)
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


def test_pointer_network_one_step_improves_loss() -> None:
    config = PointerConfig()
    params = init_params(config)
    seqs, targets = synthetic_pointer_batch()

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, seqs, targets)
    train_step_tinygrad(tinygrad_params, seqs, targets, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, seqs, targets)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, seqs, targets)
    train_step_torch(torch_params, seqs, targets, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, seqs, targets)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, seqs, targets)
    _, jax_params = train_step_jax(jax_params, seqs, targets, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, seqs, targets)
    assert float(after_jax) <= float(before_jax)
