# Notes

- SymPy status: substantive
- tinygrad status: substantive
- Cubical Agda status: minimal but meaningful
- What is superfluous here: full symbolic BPTT with dropout masks quickly becomes notation-heavy without adding much beyond the local Jacobian.
- What is still worth encoding anyway: the masked recurrence makes the regularization mechanism explicit, and the Agda layer can still encode the fact that masking preserves state type.
