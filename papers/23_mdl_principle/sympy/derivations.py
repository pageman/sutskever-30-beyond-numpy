"""Symbolic notes for MDL-style loss decomposition."""

from __future__ import annotations

from sympy import Symbol, diff, log

p = Symbol("p", positive=True)
w = Symbol("w", real=True)
lam = Symbol("lam", positive=True)
loss = -log(p) + lam * w**2
grad_w = diff(loss, w)
