"""Symbolic notes for complexity-dynamics proxy."""

from __future__ import annotations

from sympy import MatrixSymbol

x = MatrixSymbol("x", 3, 1)
W = MatrixSymbol("W", 4, 3)
h = W * x
