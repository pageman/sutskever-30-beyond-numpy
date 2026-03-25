# JAX

`impl.py` keeps the residual block functional and explicit.

This is useful because the skip-add structure shows up clearly in `lax`-based code and gives a second gradient implementation to compare against.
