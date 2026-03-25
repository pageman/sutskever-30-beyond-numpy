"""Symbolic notes for coffee automaton proxy."""

from __future__ import annotations

from sympy import MatrixSymbol

x = MatrixSymbol("state", 4, 1)
W = MatrixSymbol("W", 4, 4)
h = W * x
