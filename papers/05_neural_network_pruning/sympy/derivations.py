"""Symbolic notes for masked linear models and pruning penalties."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    w, m, x, y, lam = sp.symbols("w m x y lam", real=True)
    effective_weight = w * m
    prediction = effective_weight * x
    squared_error = (prediction - y) ** 2
    mask_penalty = lam * sp.sqrt(m**2 + 1)
    objective = sp.expand(squared_error + mask_penalty)
    grad_w = sp.simplify(sp.diff(objective, w))
    grad_m = sp.simplify(sp.diff(objective, m))
    print("objective =", objective)
    print("d objective / d w =", grad_w)
    print("d objective / d m =", grad_m)


if __name__ == "__main__":
    main()
