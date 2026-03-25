"""Symbolic notes for scaled dot-product attention."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    s1, s2, c, v1, v2, target = sp.symbols("s1 s2 c v1 v2 target", real=True)
    scale = sp.symbols("d", positive=True)
    z1 = s1 / sp.sqrt(scale)
    z2 = s2 / sp.sqrt(scale)
    p1 = sp.exp(z1) / (sp.exp(z1) + sp.exp(z2))
    p2 = sp.exp(z2) / (sp.exp(z1) + sp.exp(z2))
    shifted_p1 = sp.simplify(sp.exp(z1 + c) / (sp.exp(z1 + c) + sp.exp(z2 + c)))
    context = sp.simplify(p1 * v1 + p2 * v2)
    loss = -sp.log(p1)
    print("softmax shift invariance =", sp.simplify(shifted_p1 - p1))
    print("context =", context)
    print("d loss / d s1 =", sp.simplify(sp.diff(loss, s1)))
    print("d loss / d s2 =", sp.simplify(sp.diff(loss, s2)))


if __name__ == "__main__":
    main()
