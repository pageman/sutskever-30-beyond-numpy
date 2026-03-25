# Notes

- SymPy status: minimal
- tinygrad status: present
- Cubical Agda status: minimal
- Proxy scope: toy state-classification proxy for automaton-like transitions, not a full automata-theoretic account of the paper.
- Claim coverage: low
- Measured regime: discrete low-cardinality state encodings with deterministic labels.
- Failure modes: can overstate confidence in a classifier surrogate where the real paper interest is dynamical structure, not label prediction.
- Capability cartography note: proxy-scope framing here was informed by https://github.com/pageman/Capability-Cartography-Layer-2
- What is superfluous here: pretending this toy state-classifier is a full automata-theoretic treatment.
- What is still worth encoding anyway: discrete state encoding, transition-style interpretation, and a typed formal slice for the automaton interface.
