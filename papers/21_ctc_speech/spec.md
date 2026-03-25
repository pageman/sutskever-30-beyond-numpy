# Deep Speech 2 Spec

## Goal

Build a tiny CTC example that keeps alignment marginalization explicit without requiring a full speech stack.

## Core Equations

For a target of length `1` over `2` timesteps:

- `p(path_1) = p(blank, y)`
- `p(path_2) = p(y, blank)`
- `p(path_3) = p(y, y)`
- `p(y | x) = p(path_1) + p(path_2) + p(path_3)`
- `L = -log p(y | x)`

## Shapes

- `features`: `(B, 2, D)`
- `logits`: `(B, 2, C)`
- `targets`: `(B,)`

## Invariants

- Blank label is distinct from the target label.
- The three valid paths above are exactly the CTC alignments for this tiny setup.
- NumPy, tinygrad, PyTorch, and JAX should agree on logits and loss.

## Loss And Objective

- Primary objective: mean CTC loss on a deterministic toy batch.
- Secondary objective: one gradient step should reduce that loss.

## Minimal Deterministic Example

- `2` timesteps
- `1` target label per sample
- `3` output classes including blank

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
