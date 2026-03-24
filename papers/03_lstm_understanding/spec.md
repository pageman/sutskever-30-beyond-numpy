# Understanding LSTM Networks Spec

## Goal

Make the LSTM gating equations explicit and executable on a tiny next-character task.

## Core Equations

For concatenated input `z_t = [x_t ; h_(t-1)]`:

- `f_t = sigma(Wf z_t + bf)`
- `i_t = sigma(Wi z_t + bi)`
- `o_t = sigma(Wo z_t + bo)`
- `g_t = tanh(Wc z_t + bc)`
- `c_t = f_t * c_(t-1) + i_t * g_t`
- `h_t = o_t * tanh(c_t)`
- `y_t = Why h_t + by`

## Shapes

- `z_t`: `(V + H,)`
- `Wf, Wi, Wo, Wc`: `(H, V + H)`
- `bf, bi, bo, bc`: `(H,)`
- `Why`: `(V, H)`
- `by`: `(V,)`
- `logits`: `(T, V)`

## Invariants

- Gate activations are in `(0, 1)` for sigmoid gates.
- Cell and hidden state keep fixed shape across time.
- Torch and JAX forward passes must match the NumPy baseline from identical initial parameters.

## Loss And Objective

- Same next-token cross-entropy objective as paper 02.
- On the tiny deterministic batch, one training step should not increase the loss.

## Minimal Deterministic Example

- Text: `"hello hello world "`
- Sequence length: `5`
- Vocabulary: `"helo wrd"`
