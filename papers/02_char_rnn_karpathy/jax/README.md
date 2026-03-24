# JAX

`impl.py` mirrors the PyTorch recurrence with `jax.lax.scan`.

The point of this backend is not raw speed here. It is to force the recurrence into an explicit functional form and make Torch/JAX parity tests cheap.
