"""Symbolic notes for complexity proxy discussion."""

from __future__ import annotations

from sympy import Symbol

proxy = Symbol("proxy", real=True)
true_complexity = Symbol("K", real=True)
difference = true_complexity - proxy
