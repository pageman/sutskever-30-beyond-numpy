# Scaling Laws Spec

## Goal

Build a tiny regression model over log-parameter-count and log-loss pairs to capture the basic empirical shape behind scaling-law fits.

## Core Equations

- `x = log N`
- `y = log L`
- `y_hat = a x + b`
- `mse = mean((y_hat - y)^2)`

## Shapes

- `log parameter counts`: `(M,)`
- `log observed losses`: `(M,)`
- `predicted losses`: `(M,)`

## Invariants

- The model is linear in log-log space.
- All backends should agree on predictions and MSE.
- One step of gradient descent should improve the fit on the toy data.

## Loss And Objective

- Primary objective: mean squared error in log space.
- Secondary objective: one gradient step should lower the regression loss.

## Minimal Deterministic Example

- `4` synthetic model sizes
- closed-form power-law-like targets converted into log space

## Backend Order

- `SymPy`
- `tinygrad`
- `PyTorch`
- `JAX`
- `Cubical Agda`
