# Retrieval-Augmented Generation Spec

## Goal

Build a tiny retrieval-augmented generation toy in which retrieval weights choose among document-conditioned next-token distributions.

## Core Equations

- `q = tanh(W_q x_q + b_q)`
- `p_d = tanh(W_p x_d + b_p)`
- `w_d = softmax(q^T p_d)`
- `logits_d = W_out [q ; p_d] + b_out`
- `p(y | x) = sum_d w_d softmax(logits_d)`

## Shapes

- `queries`: `(B, D_q)`
- `documents`: `(B, K, D_p)`
- `mixture probabilities`: `(B, V)`

## Invariants

- Retrieval and generation are coupled through a weighted mixture.
- Documents compete through a normalized retrieval distribution.
- tinygrad, PyTorch, and JAX should agree on the final mixture probabilities.

## Loss And Objective

- Primary objective: cross-entropy on the retrieval-weighted token distribution.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- four queries
- three candidate documents per query
- vocabulary size `3`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
