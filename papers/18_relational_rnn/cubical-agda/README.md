# Cubical Agda

`RelationalRNN.agda` is intentionally thin.

What it captures:

- there is a memory-slot space and a recurrent output space
- slot relations map pairs of slots to a summary feature
- the recurrent step preserves memory-slot structure

What it does not try to do:

- formalize full recurrent optimization
- encode floating-point memory dynamics
- prove long-horizon properties
