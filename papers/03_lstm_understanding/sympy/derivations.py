"""Symbolic notes for LSTM gating equations."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    wf, wi, wo, wc = sp.symbols("wf wi wo wc", real=True)
    bf, bi, bo, bc = sp.symbols("bf bi bo bc", real=True)
    z, c_prev = sp.symbols("z c_prev", real=True)
    sigmoid = lambda expr: 1 / (1 + sp.exp(-expr))
    f_t = sigmoid(wf * z + bf)
    i_t = sigmoid(wi * z + bi)
    o_t = sigmoid(wo * z + bo)
    g_t = sp.tanh(wc * z + bc)
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * sp.tanh(c_t)
    print("c_t =", sp.simplify(c_t))
    print("h_t =", sp.simplify(h_t))
    print("d c_t / d c_(t-1) =", sp.simplify(sp.diff(c_t, c_prev)))


if __name__ == "__main__":
    main()
