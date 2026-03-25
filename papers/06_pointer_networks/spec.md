# Pointer Networks Spec

## Goal

Build a tiny pointer-style network that scores positions in an input sequence and predicts an index directly, rather than a token from a fixed vocabulary.

## Core Equations

- `e_i = tanh(W_enc x_i + b_enc)`
- `q = tanh(W_query mean_i x_i + b_query)`
- `s_i = v^T tanh(e_i + q)`
- `p(i | x) = softmax(s)_i`

## Shapes

- `sequence`: `(T, D)`
- `encoder states`: `(T, H)`
- `query`: `(H,)`
- `pointer logits`: `(T,)`

## Invariants

- Logits refer to positions in the source sequence.
- The scoring vector `v` is shared across all positions.
- tinygrad, PyTorch, and JAX should agree with the NumPy reference.

## Loss And Objective

- Primary objective: pointer-position cross-entropy on a tiny synthetic batch.
- Secondary objective: one gradient step should lower loss.

## Minimal Deterministic Example

- `4` positions per sequence
- item feature dimension `2`
- target pointer is one of the source indices

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
