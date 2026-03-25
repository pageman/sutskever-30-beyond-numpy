# Variational Lossy Autoencoder Spec

## Goal

Build a tiny deterministic VAE that makes the ELBO terms explicit while keeping backend parity exact.

## Core Equations

- `h = tanh(W_enc x + b_enc)`
- `mu = W_mu h + b_mu`
- `logvar = W_logvar h + b_logvar`
- `z = mu + epsilon * exp(0.5 * logvar)`
- `h_dec = tanh(W_dec z + b_dec)`
- `logits = W_out h_dec + b_out`
- `L = BCE(x, logits) + KL(q(z|x) || p(z))`

## Shapes

- `x`: `(D,)`
- `mu`: `(Z,)`
- `logvar`: `(Z,)`
- `logits`: `(D,)`

## Invariants

- The same fixed `epsilon` is used across backends to keep parity exact.
- Loss decomposes into reconstruction and KL terms.
- tinygrad, PyTorch, and JAX should agree on logits and ELBO loss.

## Loss And Objective

- Primary objective: mean ELBO on a tiny binary dataset.
- Secondary objective: one gradient step should reduce loss.

## Minimal Deterministic Example

- input dimension `4`
- latent dimension `2`
- four tiny binary observations

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
