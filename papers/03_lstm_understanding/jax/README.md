# JAX

This backend mirrors the same gate equations with `jax.lax.scan`.

The value here is structural clarity: the carried state is explicitly `(h_t, c_t)`, which lines up well with both the paper and the Agda notes.
