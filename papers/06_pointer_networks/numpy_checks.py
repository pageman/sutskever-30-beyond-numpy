"""Minimal NumPy sanity checks for Pointer Networks."""

from __future__ import annotations

from s30bn.paper06_pointer_networks import PointerConfig, forward_numpy, init_params, synthetic_pointer_batch


def main() -> None:
    config = PointerConfig()
    params = init_params(config)
    seqs, targets = synthetic_pointer_batch()
    result = forward_numpy(params, seqs, targets)
    print("paper 06 numpy loss:", round(float(result["loss"]), 6))
    print("paper 06 logits shape:", result["logits"].shape)


if __name__ == "__main__":
    main()
