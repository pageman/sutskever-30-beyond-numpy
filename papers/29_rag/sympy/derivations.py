"""Symbolic notes for retrieval-augmented generation."""

from __future__ import annotations

from sympy import Symbol, log

w1 = Symbol("w1", positive=True)
w2 = Symbol("w2", positive=True)
p1 = Symbol("p1", positive=True)
p2 = Symbol("p2", positive=True)
mixture = w1 * p1 + w2 * p2
loss = -log(mixture)
