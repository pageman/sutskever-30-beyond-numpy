# JAX

This backend reproduces the same tiny CNN with `jax.lax.conv_general_dilated`.

The useful comparison is not framework ideology. It is that both backends start from the same NumPy weights and must agree on logits and loss.
