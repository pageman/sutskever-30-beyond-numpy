# MDL Principle Spec

## Goal

Build a tiny classifier whose objective is the sum of predictive error and an explicit model-cost penalty.

## Core Equations

- `logits = W x + b`
- `L_data = CE(softmax(logits), y)`
- `L_model = lambda (mean(W^2) + mean(b^2))`
- `L_total = L_data + L_model`

## Shapes

- `input`: `(D,)`
- `class logits`: `(K,)`

## Invariants

- Predictive fit and descriptive cost remain separate terms.
- The penalty is small but always present.
- tinygrad, PyTorch, and JAX should agree with the NumPy logits and total loss.

## Loss And Objective

- Primary objective: minimize prediction error plus model cost.
- Secondary objective: one gradient step should reduce total loss.

## Minimal Deterministic Example

- input dimension `3`
- binary classification
- explicit complexity coefficient

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
