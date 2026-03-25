"""Additive-attention score notes for paper 14."""

from __future__ import annotations

from sympy import Symbol, diff, tanh

h = Symbol("h", real=True)
s = Symbol("s", real=True)
wh = Symbol("wh", real=True)
ws = Symbol("ws", real=True)
v = Symbol("v", real=True)

score = v * tanh(wh * h + ws * s)
dscore_dh = diff(score, h).simplify()

if __name__ == "__main__":
    print("score =", score)
    print("d score / d h =", dscore_dh)
