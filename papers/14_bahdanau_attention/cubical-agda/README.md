# Cubical Agda

`BahdanauAttention.agda` is intentionally thin.

What it captures:

- encoder state, decoder state, and context are distinct typed objects
- attention maps a family of encoder states and a decoder state to a context
- decoding consumes that context to produce an output symbol

What it does not try to do:

- formalize softmax numerics
- formalize full sequence training
- prove translation quality properties
