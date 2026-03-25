"""Symbolic recurrence notes for paper 04."""

from __future__ import annotations

from sympy import Symbol, diff, tanh

z = Symbol("z", real=True)
m_h = Symbol("m_h", real=True)
w_h = Symbol("w_h", real=True)
h_prev = Symbol("h_prev", real=True)

h_t = tanh(z + w_h * (m_h * h_prev))
jacobian_hidden = diff(h_t, h_prev).simplify()

if __name__ == "__main__":
    print("h_t =", h_t)
    print("d h_t / d h_prev =", jacobian_hidden)
