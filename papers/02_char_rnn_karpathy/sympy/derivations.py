"""Symbolic notes for the vanilla RNN recurrence."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    wxh, whh, bh, why, by = sp.symbols("wxh whh bh why by", real=True)
    x_t, h_prev = sp.symbols("x_t h_prev", real=True)
    h_t = sp.tanh(wxh * x_t + whh * h_prev + bh)
    y_t = why * h_t + by
    recurrence_jacobian = sp.diff(h_t, h_prev)
    print("h_t =", h_t)
    print("y_t =", y_t)
    print("d h_t / d h_(t-1) =", sp.simplify(recurrence_jacobian))


if __name__ == "__main__":
    main()
