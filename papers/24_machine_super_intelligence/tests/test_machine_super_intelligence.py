from __future__ import annotations

import numpy as np

from s30bn.paper24_machine_super_intelligence import (
    SuperIntelligenceConfig,
    forward_jax,
    forward_numpy,
    forward_tinygrad,
    forward_torch,
    init_params,
    params_to_jax,
    params_to_tinygrad,
    params_to_torch,
    synthetic_superintelligence_batch,
    train_step_jax,
    train_step_tinygrad,
    train_step_torch,
)


def test_machine_superintelligence_backends_match_numpy() -> None:
    config = SuperIntelligenceConfig()
    params = init_params(config)
    capabilities, targets = synthetic_superintelligence_batch()
    numpy_result = forward_numpy(params, capabilities, targets)
    tinygrad_loss, tinygrad_logits = forward_tinygrad(params_to_tinygrad(params), capabilities, targets)
    torch_loss, torch_logits = forward_torch(params_to_torch(params), capabilities, targets)
    jax_loss, jax_logits = forward_jax(params_to_jax(params), capabilities, targets)
    assert np.allclose(numpy_result["loss"], tinygrad_loss.item(), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(torch_loss.detach()), atol=1e-8)
    assert np.allclose(numpy_result["loss"], float(jax_loss), atol=1e-8)
    assert np.allclose(numpy_result["logits"], tinygrad_logits.numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], torch_logits.detach().numpy(), atol=1e-8)
    assert np.allclose(numpy_result["logits"], np.asarray(jax_logits), atol=1e-8)


def test_machine_superintelligence_one_step_improves_loss() -> None:
    config = SuperIntelligenceConfig()
    params = init_params(config)
    capabilities, targets = synthetic_superintelligence_batch()

    tinygrad_params = params_to_tinygrad(params)
    before_tinygrad, _ = forward_tinygrad(tinygrad_params, capabilities, targets)
    train_step_tinygrad(tinygrad_params, capabilities, targets, config.learning_rate)
    after_tinygrad, _ = forward_tinygrad(tinygrad_params, capabilities, targets)
    assert after_tinygrad.item() <= before_tinygrad.item()

    torch_params = params_to_torch(params)
    before_torch, _ = forward_torch(torch_params, capabilities, targets)
    train_step_torch(torch_params, capabilities, targets, config.learning_rate)
    after_torch, _ = forward_torch(torch_params, capabilities, targets)
    assert float(after_torch.detach()) <= float(before_torch.detach())

    jax_params = params_to_jax(params)
    before_jax, _ = forward_jax(jax_params, capabilities, targets)
    _, jax_params = train_step_jax(jax_params, capabilities, targets, config.learning_rate)
    after_jax, _ = forward_jax(jax_params, capabilities, targets)
    assert float(after_jax) <= float(before_jax)
