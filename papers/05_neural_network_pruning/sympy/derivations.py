"""Symbolic notes for sparse masking."""

from __future__ import annotations

from sympy import MatrixSymbol

W = MatrixSymbol("W", 4, 4)
M = MatrixSymbol("M", 4, 4)
effective = W.multiply_elementwise(M)
