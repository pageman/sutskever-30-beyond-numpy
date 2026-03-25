# SymPy

This paper gets a real symbolic mask-aware recurrence.

- `derivations.py` defines `h_t = tanh(Wxh (x_t m_x) + Whh (h_(t-1) m_h) + bh)` in scalar form.
- The useful symbolic object is the Jacobian with respect to the previous hidden state because the hidden-mask factor appears directly in the gradient path.
- A full unrolled symbolic derivation is intentionally omitted because the fixed-mask local Jacobian is the part that carries the explanatory value.
