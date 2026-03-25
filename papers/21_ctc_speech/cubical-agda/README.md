# Cubical Agda

`CTC.agda` is intentionally thin.

What it captures:

- there is a blank symbol distinct from emitted labels
- alignments map time steps to symbols
- collapsing an alignment yields an output sequence

What it does not try to do:

- formalize softmax numerics
- encode full beam search
- prove speech-recognition performance
