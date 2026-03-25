"""Residual-algebra notes for paper 10."""

from __future__ import annotations

from sympy import Function, Symbol, diff

x = Symbol("x", real=True)
g = Function("g")
f = x + g(x)
df_dx = diff(f, x)

if __name__ == "__main__":
    print("f(x) =", f)
    print("f'(x) =", df_dx)
