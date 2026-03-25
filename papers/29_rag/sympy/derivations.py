"""Symbolic notes for retrieval-augmented generation mixtures."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    w1, w2 = sp.symbols("w1 w2", positive=True)
    p1, p2 = sp.symbols("p1 p2", positive=True)
    mixture = sp.simplify(w1 * p1 + w2 * p2)
    loss = -sp.log(mixture)
    print("mixture =", mixture)
    print("d loss / d w1 =", sp.simplify(sp.diff(loss, w1)))
    print("d loss / d p1 =", sp.simplify(sp.diff(loss, p1)))
    print("d^2 loss / d w1 d p1 =", sp.simplify(sp.diff(loss, w1, p1)))


if __name__ == "__main__":
    main()
