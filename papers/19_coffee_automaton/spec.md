# The Coffee Automaton Spec

## Goal

Build a tiny finite-state coffee machine proxy in which discrete machine states are mapped to output classes through a simple learned readout.

## Core Equations

- `h = tanh(W_state x + b_state)`
- `logits = W_out h + b_out`

## Shapes

- `state encoding`: `(D,)`
- `hidden`: `(H,)`
- `class logits`: `(K,)`

## Invariants

- Inputs represent discrete coffee-machine states, not free-form sensory observations.
- The executable object is a toy automaton proxy, not a realistic machine controller.
- tinygrad, PyTorch, and JAX should agree with the NumPy logits.

## Loss And Objective

- Primary objective: multi-class classification over the automaton states.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- four one-hot machine states
- three output classes
- hidden dimension `4`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
