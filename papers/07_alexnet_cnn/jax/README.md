# JAX

`impl.py` expresses the same compact conv stack functionally through `lax.conv_general_dilated`.

The point here is backend parity and a second autodiff stack, not chasing XLA-specific sophistication.
