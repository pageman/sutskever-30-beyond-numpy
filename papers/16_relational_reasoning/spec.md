# Relational Reasoning Spec

## Goal

Build a compact relation network that classifies scenes by aggregating pairwise object interactions.

## Core Equations

- `g_ij = relu(W2 relu(W1 [o_i ; o_j] + b1) + b2)`
- `r = sum_{i,j} g_ij`
- `logits = Wf r + bf`

## Shapes

- `scene`: `(N_obj, D_obj)`
- `pair`: `(2 * D_obj,)`
- `g_ij`: `(H,)`
- `logits`: `(K,)`

## Invariants

- Ordered object pairs are all processed by the same `g` network.
- Aggregation is permutation-sensitive only through the ordered pair construction.
- tinygrad, PyTorch, and JAX should agree on logits for the synthetic scenes.

## Loss And Objective

- Primary objective: cross-entropy on a tiny binary relation dataset.
- Secondary objective: one step of gradient descent should reduce loss.

## Minimal Deterministic Example

- `3` objects per scene
- object feature dimension `3`
- four tiny scenes with binary relation labels

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
