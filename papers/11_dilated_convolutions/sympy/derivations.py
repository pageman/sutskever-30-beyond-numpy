"""Receptive-field arithmetic for paper 11."""

from __future__ import annotations

from sympy import Symbol, simplify

k = Symbol("k", positive=True, integer=True)
d = Symbol("d", positive=True, integer=True)
effective_width = simplify(k + (k - 1) * (d - 1))

if __name__ == "__main__":
    print("effective kernel width =", effective_width)
