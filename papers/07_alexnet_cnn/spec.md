# ImageNet Classification with Deep Convolutional Neural Networks Spec

## Goal

Build a compact AlexNet-inspired convolutional classifier that preserves the paper's main lesson: stacked learned feature extractors plus ReLU nonlinearity can separate visual classes cleanly.

## Core Equations

- `h1 = relu(conv2d(x, W1) + b1)`
- `h2 = relu(conv2d(h1, W2) + b2)`
- `g = mean_pool(h2)`
- `logits = Wf g + bf`
- `p = softmax(logits)`

Loss:

- `L = -(1 / N) sum_i log p_i[label_i]`

## Shapes

- `images`: `(N, C, H, W)`
- `conv1_w`: `(C1, C, 3, 3)`
- `conv2_w`: `(C2, C1, 3, 3)`
- `fc_w`: `(K, C2)`
- `logits`: `(N, K)`

## Invariants

- Spatial size is preserved by `padding=1`.
- Backends should agree exactly on logits for the deterministic synthetic batch.
- The model stays small enough to compare by inspection across all executable backends.

## Loss And Objective

- Primary objective: multiclass cross-entropy on a synthetic image batch with four classes.
- Secondary objective: one gradient step should lower loss on that tiny batch.

## Minimal Deterministic Example

- Image size: `12 x 12`
- Channels: `3`
- Classes: stripe, cross-stripe, diagonal, center-block
- Architecture: `conv -> relu -> conv -> relu -> global average pool -> linear`

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
