from __future__ import annotations

from s30bn.paper25_kolmogorov import KolmogorovConfig, forward_torch, init_params, params_to_torch, synthetic_kolmogorov_batch


def main() -> None:
    config = KolmogorovConfig()
    sequences, targets = synthetic_kolmogorov_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), sequences, targets)
    print("paper 25 torch loss:", round(float(loss.detach()), 6))
    print("paper 25 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
