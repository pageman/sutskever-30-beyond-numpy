"""Small symbolic notes for CS231n fundamentals."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    h, w, k, p, s = sp.symbols("h w k p s", integer=True, positive=True)
    out_h = sp.floor((h - k + 2 * p) / s) + 1
    score_j, score_t = sp.symbols("score_j score_t", real=True)
    margin = sp.Max(0, score_j - score_t + 1)
    print("conv output height =", out_h)
    print("hinge margin =", margin)


if __name__ == "__main__":
    main()
