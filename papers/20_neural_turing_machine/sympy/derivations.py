"""Content-address score notes for paper 20."""

from __future__ import annotations

from sympy import Symbol, diff

m = Symbol("m", real=True)
k = Symbol("k", real=True)
score = m * k
dscore_dk = diff(score, k)

if __name__ == "__main__":
    print("score =", score)
    print("d score / d k =", dscore_dk)
