"""Symbolic notes for position bias and middle-preference proxies."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    pos, center, gamma, content = sp.symbols("pos center gamma content", real=True)
    bias = -gamma * (pos - center) ** 2
    score = sp.simplify(content + bias)
    derivative = sp.simplify(sp.diff(score, pos))
    stationary = sp.solve(sp.Eq(derivative, 0), pos)
    print("score =", score)
    print("d score / d pos =", derivative)
    print("stationary point =", stationary)


if __name__ == "__main__":
    main()
