# The First Law of Complexodynamics Spec

## Goal

Build a tiny dynamical-state classifier that treats increasing structural richness as a signal, while remaining explicit that this is only a toy executable proxy for a broader theoretical idea.

## Core Equations

- `h = tanh(W_flow x + b_flow)`
- `logits = W_out h + b_out`

## Shapes

- `state`: `(D,)`
- `hidden`: `(H,)`
- `class logits`: `(K,)`

## Invariants

- The executable object is a toy state-evolution readout, not a full law of complexodynamics.
- Higher-structure states are separated from simpler states only within the synthetic batch.
- tinygrad, PyTorch, and JAX should agree with the NumPy logits.

## Loss And Objective

- Primary objective: binary classification over the toy state samples.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- state dimension `3`
- hidden dimension `4`
- binary classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
