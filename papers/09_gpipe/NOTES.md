# Notes

- SymPy status: minimal but present
- tinygrad status: substantive
- Cubical Agda status: minimal but meaningful
- What is superfluous here: symbolic differentiation adds less value than the scheduling idea.
- What is still worth encoding anyway: stage partitioning, ordered microbatch flow, and typed pipeline interfaces are the actual point.
- Proxy scope: a two-stage toy pipeline with a single partition boundary and shared synthetic classification objective.
- Claim coverage: captures staged forward composition and microbatch separation, not the full systems story around large-scale pipeline parallel training.
- Measured regime: deterministic small-batch loss parity, one-step improvement, and thin cross-backend gradient agreement on a synthetic pipeline example.
- Failure modes: easy to overread as validating GPipe-scale throughput or scheduling efficiency; it does not test bubble costs, distributed execution, or communication overlap.
- Capability cartography note: proxy-scope framing here was informed by https://github.com/pageman/Capability-Cartography-Layer-2
