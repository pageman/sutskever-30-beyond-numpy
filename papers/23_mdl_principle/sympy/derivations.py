"""Symbolic notes for MDL-style loss decomposition."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    p, w, lam, sigma2 = sp.symbols("p w lam sigma2", positive=True)
    description_length = -sp.log(p)
    parameter_code = lam * w**2 / (2 * sigma2)
    objective = sp.simplify(description_length + parameter_code)
    grad_w = sp.simplify(sp.diff(objective, w))
    hess_w = sp.simplify(sp.diff(grad_w, w))
    print("objective =", objective)
    print("d objective / d w =", grad_w)
    print("d^2 objective / d w^2 =", hess_w)


if __name__ == "__main__":
    main()
