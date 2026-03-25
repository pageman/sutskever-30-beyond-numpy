# Notes

- SymPy status: minimal
- tinygrad status: present
- Cubical Agda status: minimal
- What is superfluous here: a complete symbolic treatment of mixture-of-retrieval-and-generation is heavy relative to the value of the toy.
- What is still worth encoding anyway: retrieval weights, document-conditioned token distributions, and the mixture structure.
- Proxy scope: a tiny retrieval-conditioned classifier that mixes synthetic document scores with synthetic generation targets.
- Claim coverage: captures mixture-style retrieval-conditioning, not the full behavior of retrieval-augmented language generation over real corpora and token sequences.
- Measured regime: deterministic parity, thin gradient agreement, and one-step loss improvement on a small synthetic batch.
- Failure modes: easy to overread as evidence about factuality or long-form generation; it does not test generation quality, document freshness, or retrieval latency.
- Capability cartography note: proxy-scope framing here was informed by https://github.com/pageman/Capability-Cartography-Layer-2
