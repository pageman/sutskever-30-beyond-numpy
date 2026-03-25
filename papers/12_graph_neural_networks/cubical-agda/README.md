# Cubical Agda

`GraphNeuralNetworks.agda` is intentionally thin.

What it captures:

- there is a node space, message space, and graph-level output space
- message passing combines node information along edges
- graph readout consumes aggregated node information

What it does not try to do:

- formalize arbitrary graph topology algorithms
- encode floating-point optimization
- prove graph-learning guarantees
