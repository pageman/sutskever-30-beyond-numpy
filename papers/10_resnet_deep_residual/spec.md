# Deep Residual Learning for Image Recognition Spec

## Goal

Build a compact residual classifier that demonstrates the core ResNet idea: the model learns a residual function added to an identity path.

## Core Equations

- `stem = relu(conv2d(x, Ws) + bs)`
- `r1 = relu(conv2d(stem, W1) + b1)`
- `r2 = conv2d(r1, W2) + b2`
- `block = relu(stem + r2)`
- `g = mean_pool(block)`
- `logits = Wf g + bf`

## Shapes

- `images`: `(N, C, H, W)`
- `stem_w`: `(C, C, 3, 3)`
- `block1_w`: `(C, C, 3, 3)`
- `block2_w`: `(C, C, 3, 3)`
- `fc_w`: `(K, C)`

## Invariants

- The skip path is shape-preserving.
- Residual and identity branches must be addable without projection in this tiny example.
- tinygrad, PyTorch, and JAX logits should match exactly on the synthetic batch.

## Loss And Objective

- Primary objective: cross-entropy on a four-class synthetic image dataset.
- Secondary objective: one optimization step should not increase loss on that same batch.

## Minimal Deterministic Example

- Image size: `12 x 12`
- Channels: `4`
- One post-activation residual block after a learned stem
- Four hand-crafted geometric classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
