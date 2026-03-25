"""Slot relation notes for paper 18."""

from __future__ import annotations

from sympy import Symbol, diff, tanh

m1 = Symbol("m1", real=True)
m2 = Symbol("m2", real=True)
w1 = Symbol("w1", real=True)
w2 = Symbol("w2", real=True)

relation = tanh(w1 * m1 + w2 * m2)
drelation_dm1 = diff(relation, m1).simplify()

if __name__ == "__main__":
    print("relation =", relation)
    print("d relation / d m1 =", drelation_dm1)
