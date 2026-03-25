"""Pairwise relation notes for paper 16."""

from __future__ import annotations

from sympy import Symbol, diff

a = Symbol("a", real=True)
b = Symbol("b", real=True)
w1 = Symbol("w1", real=True)
w2 = Symbol("w2", real=True)
relation = w1 * a + w2 * b
drelation_da = diff(relation, a)

if __name__ == "__main__":
    print("relation =", relation)
    print("d relation / d a =", drelation_da)
