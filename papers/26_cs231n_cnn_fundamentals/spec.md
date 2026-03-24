# CS231n CNNs for Visual Recognition Spec

## Goal

Provide a tiny but real convolutional classifier that reflects the CS231n progression from pixels to logits.

## Core Equations

- `conv = Conv2D(x, W_conv, b_conv, padding=1)`
- `a = ReLU(conv)`
- `z = mean_pool(a)` over spatial dimensions
- `logits = W_fc z + b_fc`
- `loss = cross_entropy(logits, labels)`

## Shapes

- input images: `(N, C, H, W)`
- `conv_w`: `(C_out, C_in, 3, 3)`
- `conv_b`: `(C_out,)`
- pooled activations: `(N, C_out)`
- `fc_w`: `(K, C_out)`
- `fc_b`: `(K,)`
- logits: `(N, K)`

## Invariants

- Torch and JAX receive the same synthetic images and the same NumPy-initialized weights.
- Spatial shape is preserved by same-padding convolution.
- The classifier head only sees channel summaries after global mean pooling.

## Loss And Objective

- Objective: classify four deterministic synthetic image patterns.
- One training step should not increase loss on the same tiny dataset.

## Minimal Deterministic Example

- Dataset: four `8x8` RGB images
- Labels:
  - `0`: vertical stripes
  - `1`: horizontal stripes
  - `2`: diagonals
  - `3`: centered block
