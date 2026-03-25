"""Two-stage composition for paper 09."""

from __future__ import annotations

from sympy import Symbol

x = Symbol("x", real=True)
w1 = Symbol("w1", real=True)
b1 = Symbol("b1", real=True)
w2 = Symbol("w2", real=True)
b2 = Symbol("b2", real=True)
pipeline = w2 * (w1 * x + b1) + b2

if __name__ == "__main__":
    print("pipeline composition =", pipeline)
