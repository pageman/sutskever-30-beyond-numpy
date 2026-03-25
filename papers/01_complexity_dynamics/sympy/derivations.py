"""Symbolic proxy notes for complexity-dynamics-style state sensitivity."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    x1, x2, x3 = sp.symbols("x1 x2 x3", real=True)
    a1, a2, a3 = sp.symbols("a1 a2 a3", real=True)
    b = sp.symbols("b", real=True)
    state_norm = sp.sqrt(x1**2 + x2**2 + x3**2)
    score = a1 * x1 + a2 * x2 + a3 * x3 + b
    complexity_proxy = sp.log(1 + state_norm**2)
    energy = score**2 + complexity_proxy
    gradient = [sp.simplify(sp.diff(energy, var)) for var in (x1, x2, x3)]
    hessian = sp.hessian(energy, (x1, x2, x3))
    print("energy =", sp.simplify(energy))
    print("grad energy =", gradient)
    print("trace(H) =", sp.simplify(sp.trace(hessian)))


if __name__ == "__main__":
    main()
