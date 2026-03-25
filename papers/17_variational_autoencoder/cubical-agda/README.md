# Cubical Agda

`VAE.agda` is intentionally thin.

What it captures:

- there is an observed space and a latent space
- encoding maps observations to latent parameters
- decoding maps latent representations back to observations

What it does not try to do:

- formalize continuous optimization
- encode probability densities numerically
- prove generative quality claims
