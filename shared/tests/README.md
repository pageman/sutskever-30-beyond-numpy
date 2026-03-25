# Shared Tests

Repository-wide tests and reusable test modules live here.

Current contents:
- repo-level gradient parity sweeps that apply across all papers

Current design:
- paper-specific invariants and smoke tests stay under `papers/*/tests`
- cross-paper verification harnesses can live here when they are genuinely repo-wide rather than paper-local
