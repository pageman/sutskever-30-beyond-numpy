"""Minimal NumPy sanity checks for Scaling Laws."""

from __future__ import annotations

from s30bn.paper22_scaling_laws import forward_numpy, init_params, synthetic_scaling_data


def main() -> None:
    params = init_params()
    log_n, log_loss = synthetic_scaling_data()
    result = forward_numpy(params, log_n, log_loss)
    print("paper 22 numpy loss:", round(float(result["loss"]), 6))
    print("paper 22 predictions shape:", result["predictions"].shape)


if __name__ == "__main__":
    main()
