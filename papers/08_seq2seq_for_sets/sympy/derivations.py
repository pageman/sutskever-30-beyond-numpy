"""Pooling identity for paper 08."""

from __future__ import annotations

from sympy import Symbol, simplify

x1 = Symbol("x1", real=True)
x2 = Symbol("x2", real=True)
mean_pool = simplify((x1 + x2) / 2)

if __name__ == "__main__":
    print("mean pool =", mean_pool)
