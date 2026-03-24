# Cubical Agda

This layer is intentionally thin and shape-oriented.

What is worth keeping:

- images have structured channel and spatial dimensions
- a convolutional layer changes channels but preserves a valid image pipeline
- a classifier maps features to labels

What is not worth forcing here:

- low-level convolution arithmetic over numeric tensors
- optimization proofs
