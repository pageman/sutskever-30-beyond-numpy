"""Symbolic notes for dense passage retrieval."""

from __future__ import annotations

from sympy import Symbol, log

s_pos = Symbol("s_pos", real=True)
Z = Symbol("Z", positive=True)
loss = -(s_pos - log(Z))
