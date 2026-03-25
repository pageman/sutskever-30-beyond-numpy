# Order Matters Sequence to Sequence for Sets Spec

## Goal

Build a compact set encoder with a sequence-style prediction head that shows how pooled set structure can drive ordered output decisions.

## Core Equations

- `e_i = tanh(W_item x_i + b_item)`
- `p = mean_i e_i`
- `h = tanh(W_pool p + b_pool)`
- `logits = W_out h + b_out`

## Shapes

- `set`: `(N, D)`
- `embeddings`: `(N, H)`
- `logits`: `(K,)`

## Invariants

- Pooling is permutation-invariant over the set items.
- The output head is shared across all sets.
- tinygrad, PyTorch, and JAX should agree on logits.

## Loss And Objective

- Primary objective: set-level cross-entropy on a tiny synthetic batch.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- `3` items per set
- item feature dimension `2`
- three output classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
