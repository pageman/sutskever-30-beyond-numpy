# Machine Super Intelligence Spec

## Goal

Build a tiny capability aggregation model that maps several capability dimensions into a simple downstream risk classification.

## Core Equations

- `h = tanh(W_hidden x + b_hidden)`
- `logits = W_out h + b_out`

## Shapes

- `capability vector`: `(D,)`
- `hidden`: `(H,)`
- `class logits`: `(K,)`

## Invariants

- Multiple capability dimensions are aggregated through a shared representation.
- This is a toy abstraction, not a substantive intelligence model.
- tinygrad, PyTorch, and JAX should agree with the NumPy logits.

## Loss And Objective

- Primary objective: binary cross-entropy classification over the toy capability states.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- three capability dimensions
- hidden size `4`
- binary output classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
