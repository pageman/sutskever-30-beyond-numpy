from __future__ import annotations

from s30bn.paper22_scaling_laws import forward_tinygrad, init_params, params_to_tinygrad, synthetic_scaling_data


def main() -> None:
    params = params_to_tinygrad(init_params())
    log_n, log_loss = synthetic_scaling_data()
    loss, predictions = forward_tinygrad(params, log_n, log_loss)
    print("paper 22 tinygrad loss:", round(loss.item(), 6))
    print("paper 22 tinygrad predictions shape:", tuple(predictions.shape))


if __name__ == "__main__":
    main()
