# GPipe Spec

## Goal

Build a tiny two-stage network evaluated in microbatches to make pipeline partitioning explicit.

## Core Equations

- `h = tanh(W1 x + b1)`
- `logits = W2 h + b2`
- `L = cross_entropy(logits, y)`

Pipeline execution:

- split the batch into microbatches
- run stage 1 on each microbatch
- feed stage 1 outputs into stage 2
- concatenate logits in original batch order

## Shapes

- `batch`: `(B, D)`
- `microbatch`: `(B_micro, D)`
- `hidden`: `(B_micro, H)`
- `logits`: `(B, K)`

## Invariants

- Concatenated microbatch logits preserve original batch order.
- Microbatch execution should match monolithic execution for this simple two-stage network.
- tinygrad, PyTorch, and JAX should agree on logits.

## Loss And Objective

- Primary objective: cross-entropy on a deterministic toy batch.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- batch size `4`
- microbatch size `2`
- two-stage MLP split across the microbatches

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
