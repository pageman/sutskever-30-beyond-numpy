# Cubical Agda

`RNNRegularization.agda` is intentionally thin.

What it captures:

- there is a token space, hidden-state space, and mask space
- masking is an endomorphism on hidden states
- a recurrent step with masking still produces a hidden state

What it does not try to do:

- formalize stochastic dropout sampling
- formalize floating-point optimization
- unroll BPTT proofs
