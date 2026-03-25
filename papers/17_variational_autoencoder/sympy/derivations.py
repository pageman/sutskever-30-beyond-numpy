"""Scalar KL term for paper 17."""

from __future__ import annotations

from sympy import Symbol, exp, simplify

mu = Symbol("mu", real=True)
logvar = Symbol("logvar", real=True)
kl = simplify((exp(logvar) + mu**2 - 1 - logvar) / 2)

if __name__ == "__main__":
    print("KL term =", kl)
