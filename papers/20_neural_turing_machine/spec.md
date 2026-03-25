# Neural Turing Machines Spec

## Goal

Build a tiny content-addressed read/write memory model that captures the core NTM intuition without the full systems complexity.

## Core Equations

- `k_t = tanh(W_key x_t + b_key)`
- `w_t = softmax(M_t k_t)`
- `r_t = sum_i w_{t,i} M_{t,i}`
- `logits_t = W_out [r_t ; x_t] + b_out`
- `a_t = tanh(W_add x_t + b_add)`
- `M_{t+1} = M_t + w_t outer a_t`

## Shapes

- `memory`: `(N, W)`
- `weights`: `(N,)`
- `read`: `(W,)`
- `inputs`: `(T,)`
- `logits`: `(T, V)`

## Invariants

- Attention weights over memory slots sum to `1`.
- Memory slot count and width remain fixed.
- NumPy, tinygrad, PyTorch, and JAX should agree on logits.

## Loss And Objective

- Primary objective: next-token cross-entropy on a tiny deterministic sequence.
- Secondary objective: one gradient step should reduce loss on that sequence.

## Minimal Deterministic Example

- Vocabulary: `"abc "`
- Memory slots: `3`
- Memory width: `4`
- Sequence length: `3`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
