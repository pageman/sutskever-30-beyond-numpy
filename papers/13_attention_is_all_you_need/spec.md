# Attention Is All You Need Spec

## Goal

Build a tiny single-head self-attention block that shows the core Transformer mechanism without relying on the full production architecture.

## Core Equations

- `Q = X W_Q`
- `K = X W_K`
- `V = X W_V`
- `A = softmax(Q K^T / sqrt(d))`
- `C = A V`
- `logits = W_O mean_t C_t + b_O`

## Shapes

- `tokens`: `(T, D)`
- `attention scores`: `(T, T)`
- `context`: `(T, D)`
- `class logits`: `(K,)`

## Invariants

- Attention weights are normalized along the key axis.
- Query, key, and value projections share the model dimension.
- tinygrad, PyTorch, and JAX should agree on pooled logits.

## Loss And Objective

- Primary objective: sequence-level cross-entropy on a tiny synthetic classification batch.
- Secondary objective: one gradient step should lower loss.

## Minimal Deterministic Example

- `3` tokens
- model dimension `4`
- `3` output classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
