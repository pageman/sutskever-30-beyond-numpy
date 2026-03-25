"""Symbolic notes for multi-token prediction objectives."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    p1, p2 = sp.symbols("p1 p2", positive=True)
    w1, w2 = sp.symbols("w1 w2", positive=True)
    loss = -(w1 * sp.log(p1) + w2 * sp.log(p2))
    grad_p1 = sp.simplify(sp.diff(loss, p1))
    grad_p2 = sp.simplify(sp.diff(loss, p2))
    balanced = sp.simplify(loss.subs({w1: sp.Rational(1, 2), w2: sp.Rational(1, 2)}))
    print("loss =", loss)
    print("balanced loss =", balanced)
    print("d loss / d p1 =", grad_p1)
    print("d loss / d p2 =", grad_p2)


if __name__ == "__main__":
    main()
