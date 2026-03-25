# JAX

`impl.py` keeps the same deterministic masks as the other backends, but expresses the recurrence through `lax.scan`.

That separation is useful because the masking logic remains explicit while the functional update rule stays compact.
