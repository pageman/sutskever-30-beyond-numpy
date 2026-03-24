# Cubical Agda

`LSTM.agda` captures the part that is most worth formalizing:

- hidden state and cell state are distinct objects
- each gate lives in the hidden-state space
- an LSTM step preserves the joint state type

This is intentionally not a proof-heavy formalization of gradient descent.
