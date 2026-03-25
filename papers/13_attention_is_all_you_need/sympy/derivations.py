"""Symbolic notes for scaled dot-product attention."""

from __future__ import annotations

from sympy import MatrixSymbol, Symbol, log, sqrt

Q = MatrixSymbol("Q", 3, 4)
K = MatrixSymbol("K", 3, 4)
V = MatrixSymbol("V", 3, 4)
d = Symbol("d", positive=True)
scale = sqrt(d)
score_symbol = (Q * K.T) / scale
cross_entropy_term = -log(Symbol("p_y", positive=True))
