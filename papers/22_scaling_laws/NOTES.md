# Notes

- SymPy status: minimal
- tinygrad status: present
- Cubical Agda status: minimal
- What is superfluous here: elaborate symbolic manipulation is mostly unnecessary for a two-parameter log-log regression.
- What is still worth encoding anyway: the log transform, linear fit, and the distinction between empirical scaling claims and a toy executable regression.
- Proxy scope: a tiny log-log regression over synthetic loss-vs-scale points rather than a full empirical study over model and data scales.
- Claim coverage: captures the algebra of fitting a power-law line in transformed space, not the empirical universality claims of the original literature.
- Measured regime: parity, tiny gradient agreement, and single-step optimization on four deterministic synthetic observations.
- Failure modes: easy to mistake the toy for evidence about real scaling exponents; it says nothing about dataset quality, optimization dynamics, or extrapolation reliability.
- Capability cartography note: proxy-scope framing here was informed by https://github.com/pageman/Capability-Cartography-Layer-2
