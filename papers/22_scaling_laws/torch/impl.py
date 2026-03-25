from __future__ import annotations

from s30bn.paper22_scaling_laws import forward_torch, init_params, params_to_torch, synthetic_scaling_data


def main() -> None:
    params = params_to_torch(init_params())
    log_n, log_loss = synthetic_scaling_data()
    loss, predictions = forward_torch(params, log_n, log_loss)
    print("paper 22 torch loss:", round(float(loss.detach()), 6))
    print("paper 22 torch predictions shape:", tuple(predictions.shape))


if __name__ == "__main__":
    main()
