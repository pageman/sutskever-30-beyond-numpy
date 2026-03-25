# Kolmogorov Complexity Spec

## Goal

Build a tiny classifier that separates visibly compressible binary sequences from more irregular ones, while remaining honest that this is only a toy shadow of Kolmogorov complexity.

## Core Equations

- `h = tanh(W_hidden x + b_hidden)`
- `logits = W_out h + b_out`

## Shapes

- `sequence`: `(T,)`
- `hidden`: `(H,)`
- `class logits`: `(K,)`

## Invariants

- The executable object is a proxy task, not a true Kolmogorov-complexity computation.
- Repetitive and structured patterns are treated as low-complexity toy cases.
- tinygrad, PyTorch, and JAX should agree with the NumPy logits.

## Loss And Objective

- Primary objective: classify toy sequences as lower-structure-cost vs higher-irregularity.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- binary sequences of length `6`
- two classes
- intentionally tiny and interpretable dataset

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
