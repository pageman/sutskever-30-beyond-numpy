# Cubical Agda

`CharRNN.agda` is intentionally thin.

What it captures:

- there is a distinct vocabulary space and hidden-state space
- parameters map between those spaces
- a recurrent step preserves hidden-state type

What it does not try to do:

- formalize floating-point training
- formalize BPTT
- encode softmax numerics
