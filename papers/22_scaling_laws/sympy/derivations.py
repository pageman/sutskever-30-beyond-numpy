"""Symbolic notes for log-log scaling regression."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    a, b = sp.symbols("a b", real=True)
    x1, x2, y1, y2 = sp.symbols("x1 x2 y1 y2", real=True)
    objective = (a * x1 + b - y1) ** 2 + (a * x2 + b - y2) ** 2
    grad_a = sp.simplify(sp.diff(objective, a))
    grad_b = sp.simplify(sp.diff(objective, b))
    solution = sp.solve((sp.Eq(grad_a, 0), sp.Eq(grad_b, 0)), (a, b), dict=True)
    print("objective =", sp.expand(objective))
    print("d objective / d a =", grad_a)
    print("d objective / d b =", grad_b)
    print("normal-equation solution =", solution)


if __name__ == "__main__":
    main()
