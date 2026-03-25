# Cubical Agda

`Seq2SeqForSets.agda` is intentionally thin.

What it captures:

- there is a set item space and an output space
- pooling combines item representations into a summary
- decoding consumes that summary

What it does not try to do:

- formalize full sequence decoding
- prove permutation theorems in full generality
- encode optimization
