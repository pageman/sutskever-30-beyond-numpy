# Cubical Agda

`AlexNet.agda` is intentionally thin.

What it captures:

- convolutional feature extraction composes into a classifier
- image, feature, and label spaces remain distinct
- the network is a composition of typed stages

What it does not try to do:

- formalize convolution arithmetic
- reason about ImageNet-scale optimization
- encode normalization tricks from the full paper
