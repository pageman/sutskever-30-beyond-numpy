# Cubical Agda

`ResNet.agda` captures the typed residual idea directly.

What it captures:

- a residual branch is an endomorphism on feature space
- the skip path is the identity on that same space
- the block output stays in feature space

What it does not try to do:

- prove optimization properties of deep residual stacks
- formalize convolution numerics
- encode normalization layers
