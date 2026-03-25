# Neural Message Passing for Graphs Spec

## Goal

Build a compact graph classifier that makes message passing explicit at the node level.

## Core Equations

- `m_i = relu(W_msg x_i + b_msg)`
- `s_i = relu(W_self x_i + b_self)`
- `h_i = sum_j A_ij m_j + s_i`
- `g = mean_i h_i`
- `logits = W_out g + b_out`

## Shapes

- `graph`: `(N, D)`
- `adjacency`: `(N, N)`
- `messages`: `(N, H)`
- `logits`: `(K,)`

## Invariants

- Node count is fixed for the tiny synthetic dataset.
- Aggregation respects the adjacency structure.
- tinygrad, PyTorch, and JAX should agree on logits.

## Loss And Objective

- Primary objective: graph-level cross-entropy on a tiny synthetic dataset.
- Secondary objective: one step of gradient descent should reduce loss.

## Minimal Deterministic Example

- `3` nodes per graph
- node feature dimension `2`
- binary graph classification

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
