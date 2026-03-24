# SymPy

This paper benefits from symbolic treatment more than most architecture papers.

- `derivations.py` writes out the forget, input, output, and candidate equations.
- The most useful scalar identity here is `d c_t / d c_(t-1) = f_t`, which makes the memory path explicit.
- Full sequence-level symbolic differentiation is omitted for the same reason as paper 02: it obscures the main point.
