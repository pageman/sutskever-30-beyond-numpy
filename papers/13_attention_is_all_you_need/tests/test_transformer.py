from __future__ import annotations

import jax
import numpy as np
import torch

from s30bn.paper13_transformer import (
    TransformerConfig,
    attention_weights_numpy,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_attention_batch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)
from s30bn.test_support import (
    assert_allclose_named,
    assert_array_shape,
    assert_parameter_dict_allclose,
    assert_probabilities_normalized,
    mapping_values_to_numpy,
)


def test_transformer_backends_match_numpy() -> None:
    config = TransformerConfig()
    params = init_params(config)
    seqs, targets = synthetic_attention_batch()
    numpy_result = forward_numpy(params, seqs, targets)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), seqs, targets)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), seqs, targets)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), seqs, targets)
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
    assert_allclose_named(
        tinygrad_logits.numpy(),
        {
            "torch": torch_logits.detach().numpy(),
            "jax": np.asarray(jax_logits),
        },
    )
    assert_array_shape(np.asarray(numpy_result["logits"]), (seqs.shape[0], config.num_classes))


def test_transformer_one_step_improves_loss() -> None:
    config = TransformerConfig()
    params = init_params(config)
    seqs, targets = synthetic_attention_batch()

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


def test_transformer_attention_weights_are_normalized() -> None:
    config = TransformerConfig()
    params = init_params(config)
    seqs, _ = synthetic_attention_batch()
    weights = attention_weights_numpy(params, seqs)
    assert_array_shape(weights, (seqs.shape[0], config.seq_len, config.seq_len))
    assert_probabilities_normalized(weights, axis=2)


def test_transformer_torch_and_jax_gradients_and_updated_params_match() -> None:
    config = TransformerConfig()
    params = init_params(config)
    seqs, targets = synthetic_attention_batch()

    torch_params = params_to_torch(params)
    torch_loss, _ = forward_torch(torch_params, seqs, targets)
    torch_loss.backward()
    torch_grads = mapping_values_to_numpy(torch_params, grad=True)

    jax_params = params_to_jax(params)
    jax_grads = jax.grad(lambda current: forward_jax(current, seqs, targets)[0])(jax_params)
    assert_parameter_dict_allclose(torch_grads, {"jax": jax_grads}, atol=1e-6)

    with torch.no_grad():
        for tensor in torch_params.values():
            tensor -= config.learning_rate * tensor.grad
            tensor.grad.zero_()
    _, jax_updated = train_step_jax(jax_params, seqs, targets, config.learning_rate)

    assert_parameter_dict_allclose(mapping_values_to_numpy(torch_params), {"jax": jax_updated}, atol=1e-6)

    _, torch_logits = forward_torch(torch_params, seqs, targets)
    _, jax_logits = forward_jax(jax_updated, seqs, targets)
    assert np.allclose(torch_logits.detach().numpy(), np.asarray(jax_logits), atol=1e-6)
