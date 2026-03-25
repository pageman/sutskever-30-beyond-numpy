# {{paper_id}} {{paper_title}}

## Purpose

Summarize the core claim, mathematical object, and what this repository should make concrete beyond the NumPy baseline.

## Deliverables

- `spec.md` with equations, tensor shapes, invariants, and expected outputs
- `numpy_checks.py` with tiny deterministic sanity checks
- `sympy/` with symbolic derivations or a note on why symbolic work is thin
- `tinygrad/` as the first minimal executable autodiff backend
- `torch/` as the primary executable implementation
- `jax/` as the functional executable implementation
- `cubical-agda/` with a meaningful formal slice or an explicit note that the layer is intentionally minimal
- `tests/` with parity, shape, and smoke tests

## Notes

- Keep `SymPy` and `Cubical Agda` present for completeness even when they are not the most informative backend for the paper.
- Keep `tinygrad` present for completeness even when it is intentionally thin.
- Record omissions explicitly rather than inflating ceremonial code.
