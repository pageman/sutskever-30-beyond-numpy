# SymPy

This paper gets a real symbolic recurrence rather than a ceremonial placeholder.

- `derivations.py` encodes the scalar recurrence `h_t = tanh(Wxh x_t + Whh h_(t-1) + bh)`.
- The useful symbolic object is the Jacobian of `h_t` with respect to `h_(t-1)`, because it explains vanishing and stable gradient regimes.
- A full symbolic BPTT derivation is intentionally omitted because it becomes unreadable faster than it becomes useful.
