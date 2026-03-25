# Multi-token Prediction Spec

## Goal

Build a tiny shared-trunk model that predicts two future tokens at once, illustrating the core idea of parallel multi-token supervision.

## Core Equations

- `h = tanh(W_hidden x + b_hidden)`
- `logits_t = W_out[t] h + b_out[t]`
- `p(y_t | x) = softmax(logits_t)`
- `loss = mean_t CE(p(y_t | x), y_t)`

## Shapes

- `context`: `(D,)`
- `hidden`: `(H,)`
- `token logits`: `(T, V)`

## Invariants

- The hidden representation is shared across all predicted token positions.
- Each token slot has its own output head.
- tinygrad, PyTorch, and JAX should agree with the NumPy reference.

## Loss And Objective

- Primary objective: average cross-entropy across the predicted token slots.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- context feature dimension `3`
- two output tokens
- vocabulary size `4`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
