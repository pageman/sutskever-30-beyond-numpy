"""Symbolic notes for a Kolmogorov/MDL-style proxy gap."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    proxy, complexity, penalty = sp.symbols("proxy complexity penalty", positive=True)
    gap = complexity - proxy
    regret = sp.simplify(gap + sp.log(1 + penalty * gap))
    derivative = sp.simplify(sp.diff(regret, gap))
    second_derivative = sp.simplify(sp.diff(regret, gap, 2))
    print("regret proxy =", regret)
    print("d regret / d gap =", derivative)
    print("d^2 regret / d gap^2 =", second_derivative)


if __name__ == "__main__":
    main()
