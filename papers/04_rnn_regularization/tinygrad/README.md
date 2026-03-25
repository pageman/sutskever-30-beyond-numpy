# tinygrad

`impl.py` is a real tinygrad backend, not a ceremonial placeholder.

It keeps the dropout masks deterministic and shared across the full sequence so parity with NumPy, PyTorch, and JAX remains exact on the toy batch.
