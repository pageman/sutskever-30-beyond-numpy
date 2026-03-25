# Dense Passage Retrieval Spec

## Goal

Build a tiny dual-encoder retrieval model with in-batch negatives to capture the core DPR training pattern.

## Core Equations

- `q = tanh(W_q x_q + b_q)`
- `p = tanh(W_p x_p + b_p)`
- `score(i, j) = q_i^T p_j`
- `p(j | i) = softmax(score(i, :))_j`

## Shapes

- `queries`: `(B, D)`
- `passages`: `(B, D)`
- `score matrix`: `(B, B)`

## Invariants

- Query and passage encoders are separate.
- In-batch negatives arise from the off-diagonal scores.
- tinygrad, PyTorch, and JAX should agree with the NumPy score matrix.

## Loss And Objective

- Primary objective: cross-entropy over the in-batch positive index.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- four queries
- four passages
- positives on the diagonal

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
