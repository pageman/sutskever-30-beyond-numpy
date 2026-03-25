# Lost in the Middle Spec

## Goal

Build a tiny positional retrieval/readout model that makes position bias explicit and lets the “middle gets downweighted” effect appear in a controlled toy setting.

## Core Equations

- `h_i = tanh(W_chunk c_i + b_chunk)`
- `q = tanh(W_query x + b_query)`
- `score_i = h_i^T q + bias_i`
- `w = softmax(score)`
- `pooled = sum_i w_i h_i`
- `logits = W_out pooled + b_out`

## Shapes

- `chunks`: `(T, D)`
- `query`: `(D,)`
- `attention weights`: `(T,)`
- `class logits`: `(K,)`

## Invariants

- Position bias is explicit rather than hidden.
- Content score and position score are separable.
- tinygrad, PyTorch, and JAX should agree on the final logits.

## Loss And Objective

- Primary objective: cross-entropy on the pooled readout.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- `5` chunks per context
- middle bias initialized below edge bias
- binary answer classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
