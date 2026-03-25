"""Symbolic notes for paper 07."""

from __future__ import annotations

from sympy import Max, Symbol, diff

x = Symbol("x", real=True)
w1 = Symbol("w1", real=True)
b1 = Symbol("b1", real=True)
w2 = Symbol("w2", real=True)
b2 = Symbol("b2", real=True)

h1 = Max(0, w1 * x + b1)
h2 = Max(0, w2 * h1 + b2)
dh2_dx = diff(h2, x)

if __name__ == "__main__":
    print("h2 =", h2)
    print("d h2 / d x =", dh2_dx)
