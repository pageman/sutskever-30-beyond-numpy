"""Symbolic notes for the tiny pointer-network score."""

from __future__ import annotations

from sympy import MatrixSymbol, Symbol, exp, log, simplify

t = Symbol("t", positive=True)
score_t = Symbol("s_t", real=True)
z = MatrixSymbol("z", 1, 4)

softmax_t = exp(score_t) / Symbol("Z", positive=True)
negative_log_likelihood_t = -log(softmax_t)
pointer_attention_note = simplify(negative_log_likelihood_t)
