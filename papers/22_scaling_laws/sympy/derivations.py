"""Symbolic notes for log-log scaling regression."""

from __future__ import annotations

from sympy import Symbol, diff

a = Symbol("a", real=True)
b = Symbol("b", real=True)
x = Symbol("x", real=True)
y = Symbol("y", real=True)

y_hat = a * x + b
squared_error = (y_hat - y) ** 2
grad_a = diff(squared_error, a)
grad_b = diff(squared_error, b)
