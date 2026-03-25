"""Symbolic notes for dense passage retrieval as a softmax ranking loss."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    s_pos, s_neg1, s_neg2 = sp.symbols("s_pos s_neg1 s_neg2", real=True)
    Z = sp.exp(s_pos) + sp.exp(s_neg1) + sp.exp(s_neg2)
    loss = sp.simplify(-(s_pos - sp.log(Z)))
    print("loss =", loss)
    print("d loss / d s_pos =", sp.simplify(sp.diff(loss, s_pos)))
    print("d loss / d s_neg1 =", sp.simplify(sp.diff(loss, s_neg1)))
    print("d loss / d s_neg2 =", sp.simplify(sp.diff(loss, s_neg2)))


if __name__ == "__main__":
    main()
