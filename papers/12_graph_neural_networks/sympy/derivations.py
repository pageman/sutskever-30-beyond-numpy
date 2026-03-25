"""Message aggregation notes for paper 12."""

from __future__ import annotations

from sympy import Symbol, diff

a12 = Symbol("a12", real=True)
m2 = Symbol("m2", real=True)
self_term = Symbol("s1", real=True)
h1 = a12 * m2 + self_term
dh1_dm2 = diff(h1, m2)

if __name__ == "__main__":
    print("h1 =", h1)
    print("d h1 / d m2 =", dh1_dm2)
