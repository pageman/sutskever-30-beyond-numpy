from __future__ import annotations

from s30bn.paper30_lost_in_middle import LostMiddleConfig, forward_torch, init_params, params_to_torch, synthetic_middle_batch


def main() -> None:
    config = LostMiddleConfig()
    chunks, queries, targets = synthetic_middle_batch()
    loss, logits = forward_torch(params_to_torch(init_params(config)), chunks, queries, targets)
    print("paper 30 torch loss:", round(float(loss.detach()), 6))
    print("paper 30 torch logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
