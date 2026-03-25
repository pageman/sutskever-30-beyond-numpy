# Identity Mappings in Deep Residual Networks Spec

## Goal

Build a compact pre-activation residual block to emphasize the paper's main thesis: the skip path should remain as close to identity as possible while residual transformations do the learning.

## Core Equations

- `u = relu(x)`
- `r1 = relu(conv2d(u, W1) + b1)`
- `r2 = conv2d(r1, W2) + b2`
- `block = x + r2`
- `g = mean_pool(relu(block))`
- `logits = Wf g + bf`

## Shapes

- `images`: `(N, C, H, W)`
- `conv1_w`: `(C, C, 3, 3)`
- `conv2_w`: `(C, C, 3, 3)`
- `fc_w`: `(K, C)`

## Invariants

- The residual branch is shape-preserving, so the skip path remains a true identity.
- When residual weights are all zero, the block reduces to the identity path before the final head.
- tinygrad, PyTorch, and JAX should agree on logits for the same parameters.

## Loss And Objective

- Primary objective: cross-entropy on a synthetic four-class image batch.
- Secondary objective: one gradient step should reduce loss on that batch.
- Structural objective: zero residual weights should make the block behave like an identity map.

## Minimal Deterministic Example

- Image size: `12 x 12`
- Channels: `4`
- Pre-activation residual block followed by a linear head
- Four geometric synthetic classes

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
