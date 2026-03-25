# Relational RNNs Spec

## Goal

Build a tiny recurrent model whose state is mediated by pairwise relations between memory slots.

## Core Equations

- `r_ij = tanh(W_rel [m_i ; m_j] + b_rel)`
- `r = mean_{i,j} r_ij`
- `m'_0 = tanh(W_mem mean(m) + W_x x_t + r + b_h)`
- `m'_1 = 0.5 * (m_0 + m'_0)`
- `logits_t = W_out mean(m') + b_out`

## Shapes

- `memory`: `(S, H)`
- `r_ij`: `(H,)`
- `inputs`: `(T,)`
- `logits`: `(T, V)`

## Invariants

- Memory slot count stays fixed across time.
- Relational summary is computed from all ordered slot pairs.
- NumPy, tinygrad, PyTorch, and JAX logits should agree.

## Loss And Objective

- Primary objective: next-token cross-entropy on a tiny deterministic corpus.
- Secondary objective: one step of gradient descent should not increase loss on the same batch.

## Minimal Deterministic Example

- Vocabulary: `"helo wrd"`
- Sequence length: `4`
- Memory slots: `2`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
