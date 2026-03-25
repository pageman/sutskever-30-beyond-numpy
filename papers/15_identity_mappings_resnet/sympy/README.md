# SymPy

The symbolic layer focuses on the identity-plus-residual decomposition.

- `derivations.py` differentiates `f(x) = x + g(relu(x))`.
- The important point is that the derivative preserves an explicit identity term rather than hiding it inside a monolithic block.
