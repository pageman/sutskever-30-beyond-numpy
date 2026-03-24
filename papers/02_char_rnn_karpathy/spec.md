# The Unreasonable Effectiveness of Recurrent Neural Networks Spec

## Goal

Build a tiny character-level language model that predicts the next character in a short sequence.

## Core Equations

For each time step `t`:

- `x_t in R^V` is a one-hot character vector
- `h_t = tanh(Wxh x_t + Whh h_(t-1) + bh)`
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

## Invariants

- Each input token index is in `[0, V)`.
- Hidden state shape is constant across time.
- Torch and JAX logits must match from the same NumPy-initialized parameters.

## Loss And Objective

- Primary objective: next-token cross-entropy on a deterministic toy text.
- Secondary objective: one SGD step should not increase loss on the same batch.

## Minimal Deterministic Example

- Text: `"hello world hello world "`
- Sequence length: `5`
- Vocabulary: `"helo wrd"`
- Tiny batch: first five characters and their next-character targets
