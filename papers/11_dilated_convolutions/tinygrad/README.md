# tinygrad

`impl.py` is a real tinygrad backend.

The implementation makes dilation explicit by expanding the kernel with zeros, which keeps tinygrad aligned with the other backends without relying on backend-specific dilation support.
