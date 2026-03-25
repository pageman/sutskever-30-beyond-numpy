# Notes

- SymPy status: partial
- tinygrad status: present
- Cubical Agda status: minimal
- What is superfluous here: a complete symbolic treatment of context-position effects would be much larger than the toy warrants.
- What is still worth encoding anyway: explicit position bias, normalized attention weights, and the separation of content relevance from placement.
- Proxy scope: a small context-position toy with synthetic chunks, explicit position bias, and a single-step answer target.
- Claim coverage: captures the idea that placement affects access weight, not the full evaluation methodology behind long-context retrieval and answer accuracy studies.
- Measured regime: parity, thin gradient agreement, and position-biased loss behavior on deterministic synthetic examples.
- Failure modes: easy to overread as a full long-context benchmark; it does not test realistic prompts, model scaling, or robustness across large context windows.
- Capability cartography note: proxy-scope framing here was informed by https://github.com/pageman/Capability-Cartography-Layer-2
