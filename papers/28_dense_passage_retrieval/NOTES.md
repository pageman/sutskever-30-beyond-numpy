# Notes

- SymPy status: minimal
- tinygrad status: substantive
- Cubical Agda status: minimal
- What is superfluous here: elaborate symbolic treatment of in-batch negatives is less useful than just stating the score matrix and the diagonal target structure.
- What is still worth encoding anyway: dual encoders, contrastive scores, and the diagonal retrieval target.
- Proxy scope: a tiny dual-encoder retrieval toy with synthetic query/passage vectors and diagonal positive targets.
- Claim coverage: captures score-matrix contrastive retrieval structure, not corpus-scale indexing, hard-negative mining, or retrieval quality in realistic search settings.
- Measured regime: deterministic parity, thin gradient agreement, and one-step improvement on in-memory synthetic batches.
- Failure modes: can be overread as validating retrieval effectiveness; it does not test recall@k, ANN search, passage chunking, or large-corpus behavior.
- Capability cartography note: proxy-scope framing here was informed by https://github.com/pageman/Capability-Cartography-Layer-2
