"""Symbolic notes for positional context weighting."""

from __future__ import annotations

from sympy import Symbol

content = Symbol("content", real=True)
bias = Symbol("bias", real=True)
score = content + bias
