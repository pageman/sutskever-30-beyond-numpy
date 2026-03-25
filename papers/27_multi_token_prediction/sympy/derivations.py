"""Symbolic notes for multi-token prediction."""

from __future__ import annotations

from sympy import Symbol, log

p1 = Symbol("p1", positive=True)
p2 = Symbol("p2", positive=True)
loss = -(log(p1) + log(p2)) / 2
