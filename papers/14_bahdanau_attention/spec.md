# Neural Machine Translation by Jointly Learning to Align and Translate Spec

## Goal

Build a tiny additive-attention example that keeps the Bahdanau alignment mechanism explicit.

## Core Equations

- `h_i = tanh(W_enc x_i + b_enc)`
- `s = tanh(W_dec y_prev + b_dec)`
- `e_i = v^T tanh(W_h h_i + W_s s + b_attn)`
- `alpha = softmax(e)`
- `c = sum_i alpha_i h_i`
- `logits = W_out [c ; s] + b_out`

## Shapes

- `source_tokens`: `(T,)`
- `encoder_states`: `(T, H)`
- `decoder_state`: `(H,)`
- `attention`: `(T,)`
- `logits`: `(V,)`

## Invariants

- Attention weights must sum to `1`.
- The context vector is a convex combination of encoder states.
- NumPy, tinygrad, PyTorch, and JAX should agree on logits and attention weights.

## Loss And Objective

- Primary objective: single-step cross-entropy on a deterministic toy translation pair.
- Secondary objective: one gradient step should reduce loss on that same example.

## Minimal Deterministic Example

- Source length: `4`
- Vocabulary: `"abcd "`
- Decoder prompt token: space
- One additive-attention decode step

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
