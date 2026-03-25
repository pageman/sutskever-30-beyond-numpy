# Cubical Agda

`GPipe.agda` is intentionally thin.

What it captures:

- there are distinct pipeline stages
- stage outputs feed later stages
- microbatch order is preserved through the pipeline

What it does not try to do:

- formalize distributed scheduling
- encode actual device placement
- prove performance claims
