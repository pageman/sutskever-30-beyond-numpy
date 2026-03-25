"""Symbolic notes for capability aggregation and diminishing returns."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    c1, c2, alpha = sp.symbols("c1 c2 alpha", positive=True)
    aggregate = sp.log(1 + alpha * (c1 + c2))
    marginal_c1 = sp.simplify(sp.diff(aggregate, c1))
    marginal_c2 = sp.simplify(sp.diff(aggregate, c2))
    cross_term = sp.simplify(sp.diff(aggregate, c1, c2))
    print("aggregate capability =", aggregate)
    print("d aggregate / d c1 =", marginal_c1)
    print("d aggregate / d c2 =", marginal_c2)
    print("cross sensitivity =", cross_term)


if __name__ == "__main__":
    main()
