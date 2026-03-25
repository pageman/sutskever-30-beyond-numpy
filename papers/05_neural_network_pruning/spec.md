# Keeping Neural Networks Simple Spec

## Goal

Build a tiny sparse MLP in which a fixed pruning mask removes part of the hidden connectivity, making the simplicity bias explicit.

## Core Equations

- `W_sparse = W_hidden * mask`
- `h = tanh(W_sparse x + b_hidden)`
- `logits = W_out h + b_out`

## Shapes

- `input`: `(D,)`
- `hidden`: `(H,)`
- `class logits`: `(K,)`

## Invariants

- The pruning mask is fixed and not updated by gradient descent.
- Only the unmasked weights contribute to the hidden representation.
- tinygrad, PyTorch, and JAX should agree with the NumPy logits.

## Loss And Objective

- Primary objective: class cross-entropy under sparse connectivity.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- input dimension `4`
- hidden dimension `4`
- binary classification

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
