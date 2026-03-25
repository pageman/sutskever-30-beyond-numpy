from __future__ import annotations

import numpy as np
from s30bn.test_support import assert_array_shape, assert_loss_trajectories_close, assert_probabilities_normalized

from s30bn.paper14_bahdanau_attention import (
    BahdanauConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    sample_pair,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_bahdanau_backends_match_numpy() -> None:
    config = BahdanauConfig()
    params = init_params(config)
    source, decoder_token, target = sample_pair(config)
    numpy_result = forward_numpy(params, source, decoder_token, target, config)
    tinygrad_loss, tinygrad_logits, tinygrad_attention = forward_tinygrad(params_to_tinygrad(params), source, decoder_token, target, config)
    torch_loss, torch_logits, torch_attention = forward_torch(params_to_torch(params), source, decoder_token, target, config)
    jax_loss, jax_logits, jax_attention = forward_jax(params_to_jax(params), source, decoder_token, target, config)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["logits"], tinygrad_logits.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], torch_logits.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], np.asarray(jax_logits), atol=1e-8)
    assert np.allclose(numpy_result["attention"], tinygrad_attention.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["attention"], torch_attention.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["attention"], np.asarray(jax_attention), atol=1e-8)
    expected_logits_shape = np.asarray(numpy_result["logits"]).shape
    expected_attention_shape = np.asarray(numpy_result["attention"]).shape
    assert_array_shape(tinygrad_logits.numpy(), expected_logits_shape)
    assert_array_shape(torch_logits.detach().numpy(), expected_logits_shape)
    assert_array_shape(np.asarray(jax_logits), expected_logits_shape)
    assert_array_shape(tinygrad_attention.numpy(), expected_attention_shape)
    assert_array_shape(torch_attention.detach().numpy(), expected_attention_shape)
    assert_array_shape(np.asarray(jax_attention), expected_attention_shape)
    assert_probabilities_normalized(np.asarray(jax_attention), axis=0)


def test_bahdanau_one_step_improves_loss() -> None:
    config = BahdanauConfig()
    params = init_params(config)
    source, decoder_token, target = sample_pair(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _, _ = forward_tinygrad(tinygrad_params, source, decoder_token, target, config)
    train_step_tinygrad(tinygrad_params, source, decoder_token, target, config)
    after_tinygrad, _, _ = forward_tinygrad(tinygrad_params, source, decoder_token, target, config)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _, _ = forward_torch(torch_params, source, decoder_token, target, config)
    train_step_torch(torch_params, source, decoder_token, target, config)
    after_torch, _, _ = forward_torch(torch_params, source, decoder_token, target, config)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _, _ = forward_jax(jax_params, source, decoder_token, target, config)
    _, jax_params = train_step_jax(jax_params, source, decoder_token, target, config)
    after_jax, _, _ = forward_jax(jax_params, source, decoder_token, target, config)
    assert float(after_jax) <= float(before_jax)


def test_bahdanau_one_step_losses_match_across_backends() -> None:
    config = BahdanauConfig()
    params = init_params(config)
    source, decoder_token, target = sample_pair(config)

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _, _ = forward_tinygrad(tinygrad_params, source, decoder_token, target, config)
    train_step_tinygrad(tinygrad_params, source, decoder_token, target, config)
    after_tinygrad, _, _ = forward_tinygrad(tinygrad_params, source, decoder_token, target, config)

    torch_params = params_to_torch(params)
    before_torch, _, _ = forward_torch(torch_params, source, decoder_token, target, config)
    train_step_torch(torch_params, source, decoder_token, target, config)
    after_torch, _, _ = forward_torch(torch_params, source, decoder_token, target, config)

    jax_params = params_to_jax(params)
    before_jax, _, _ = forward_jax(jax_params, source, decoder_token, target, config)
    _, jax_params = train_step_jax(jax_params, source, decoder_token, target, config)
    after_jax, _, _ = forward_jax(jax_params, source, decoder_token, target, config)

    assert_loss_trajectories_close(
        (float(before_torch.detach()), float(after_torch.detach())),
        {
            "tinygrad": (before_tinygrad.item(), after_tinygrad.item()),
            "jax": (float(before_jax), float(after_jax)),
        },
    )
