# Dilated Convolutions Spec

## Goal

Build a compact classifier that keeps the core idea of dilated convolutions explicit: enlarge receptive field without pooling away resolution.

## Core Equations

- `h1 = relu(conv2d(x, W1) + b1)`
- `h2 = relu(dilated_conv2d(h1, Wd, dilation=2) + bd)`
- `g = mean_pool(h2)`
- `logits = Wf g + bf`

## Shapes

- `images`: `(N, C, H, W)`
- `conv1_w`: `(C1, C, 3, 3)`
- `dilated_w`: `(C2, C1, 3, 3)`
- `fc_w`: `(K, C2)`
- `logits`: `(N, K)`

## Invariants

- The dilated kernel increases effective receptive field while preserving output resolution.
- The educational implementation uses explicit zero-inserted kernels so all backends remain exactly aligned.
- tinygrad, PyTorch, and JAX logits should match on the same synthetic batch.

## Loss And Objective

- Primary objective: cross-entropy on a tiny structured image dataset.
- Secondary objective: one gradient step should reduce loss on that dataset.

## Minimal Deterministic Example

- Image size: `12 x 12`
- Channels: `2`
- Dilation factor: `2`
- Architecture: `conv -> relu -> dilated conv -> relu -> global average pool -> linear`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
