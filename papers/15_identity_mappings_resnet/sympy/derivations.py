"""Identity-mapping notes for paper 15."""

from __future__ import annotations

from sympy import Function, Max, Symbol, diff

x = Symbol("x", real=True)
g = Function("g")
f = x + g(Max(0, x))
df_dx = diff(f, x)

if __name__ == "__main__":
    print("f(x) =", f)
    print("f'(x) =", df_dx)
