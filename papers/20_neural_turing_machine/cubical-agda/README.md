# Cubical Agda

`NeuralTuringMachine.agda` is intentionally thin.

What it captures:

- there is a memory space, an addressing map, and a readout
- addressing maps controller state to memory interaction
- read/write operations preserve the existence of a memory state

What it does not try to do:

- formalize softmax numerics
- encode all NTM addressing refinements
- prove training behavior
