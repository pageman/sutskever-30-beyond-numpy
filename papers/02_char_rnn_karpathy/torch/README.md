# PyTorch

`impl.py` is the primary executable reference for this paper in the new repo.

It uses:

- one-hot character inputs
- a manually parameterized vanilla RNN cell
- cross-entropy over the next-token targets

This is intentionally small enough to compare directly against JAX.
