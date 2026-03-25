# Cubical Agda

`IdentityMappings.agda` records the paper's main conceptual invariant.

What it captures:

- the skip path is an actual identity map on feature space
- the residual branch is a separate endomorphism
- composing them stays in the same feature space

What it does not try to do:

- formalize batch normalization
- prove optimization results for very deep nets
- encode full conv arithmetic
