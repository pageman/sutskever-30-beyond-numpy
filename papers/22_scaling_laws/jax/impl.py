from __future__ import annotations

from s30bn.paper22_scaling_laws import forward_jax, init_params, params_to_jax, synthetic_scaling_data


def main() -> None:
    params = params_to_jax(init_params())
    log_n, log_loss = synthetic_scaling_data()
    loss, predictions = forward_jax(params, log_n, log_loss)
    print("paper 22 jax loss:", round(float(loss), 6))
    print("paper 22 jax predictions shape:", tuple(predictions.shape))


if __name__ == "__main__":
    main()
