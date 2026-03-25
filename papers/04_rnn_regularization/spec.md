# Recurrent Neural Network Regularization Spec

## Goal

Show how deterministic input and hidden-state dropout change a tiny next-character RNN without changing the underlying recurrence structure.

## Core Equations

For each time step `t`:

- `x_t in R^V` is a one-hot token.
- `m_x in R^V` is a fixed input dropout mask.
- `m_h in R^H` is a fixed recurrent dropout mask.
- `h_t = tanh(Wxh (x_t * m_x) + Whh (h_(t-1) * m_h) + bh)`
- `y_t = Why h_t + by`
- `p_t = softmax(y_t)`

Sequence loss:

- `L = -(1 / T) sum_t log p_t[target_t]`

## Shapes

- `V`: vocabulary size
- `H`: hidden size
- `Wxh`: `(H, V)`
- `Whh`: `(H, H)`
- `Why`: `(V, H)`
- `bh`: `(H,)`
- `by`: `(V,)`
- `inputs`: `(T,)`
- `logits`: `(T, V)`
- `m_x`: `(V,)`
- `m_h`: `(H,)`

## Invariants

- The dropout masks are deterministic and shared across the whole short sequence.
- Hidden-state shape is constant across time.
- NumPy, tinygrad, PyTorch, and JAX logits should agree from the same parameters and masks.

## Loss And Objective

- Primary objective: next-token cross-entropy on a deterministic toy corpus.
- Secondary objective: one gradient step with fixed masks should not increase loss on that same tiny batch.

## Minimal Deterministic Example

- Text: `"hello world hello "`
- Sequence length: `6`
- Vocabulary: `"helo wrd"`
- Input keep probability: `0.75`
- Hidden keep probability: `0.8`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
