"""Tiny CTC alignment sum for paper 21."""

from __future__ import annotations

from sympy import Symbol, simplify

p0b = Symbol("p0b", positive=True)
p0y = Symbol("p0y", positive=True)
p1b = Symbol("p1b", positive=True)
p1y = Symbol("p1y", positive=True)

ctc_prob = simplify(p0b * p1y + p0y * p1b + p0y * p1y)

if __name__ == "__main__":
    print("ctc probability =", ctc_prob)
