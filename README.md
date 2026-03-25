# Sutskever 30 Beyond NumPy

Multi-backend implementations of the 30 papers from the Sutskever reading list, using the canonical NumPy repo as the baseline reference and specification.

Reference baseline:
- [`pageman/sutskever-30-implementations`](https://github.com/pageman/sutskever-30-implementations)

This repository keeps the original paper numbering for compatibility, but tracks a separate `build_order` so implementation work can follow dependencies rather than the original list order.

## About

Based on the Numpy-version-only:
- https://github.com/pageman/sutskever-30-implementations

## Tooling

- Python project metadata: [`pyproject.toml`](/Users/hifi/sutskever-30-beyond-numpy/pyproject.toml)
- Demo runner: [`scripts/run_paper.py`](/Users/hifi/sutskever-30-beyond-numpy/scripts/run_paper.py)
- tinygrad installer helper: [`scripts/install_tinygrad.sh`](/Users/hifi/sutskever-30-beyond-numpy/scripts/install_tinygrad.sh)
- Agda library file: [`sutskever-30-beyond-numpy.agda-lib`](/Users/hifi/sutskever-30-beyond-numpy/sutskever-30-beyond-numpy.agda-lib)
- Agda setup notes: [`docs/AGDA_SETUP.md`](/Users/hifi/sutskever-30-beyond-numpy/docs/AGDA_SETUP.md)

## Backend Policy

Per paper, the expected implementation pipeline is:

1. `spec.md`
2. `numpy_checks.py`
3. `sympy/`
4. `tinygrad/`
5. `torch/`
6. `jax/`
7. `cubical-agda/`

Rules:
- `NumPy` is minimal and exists only for sanity checks, fixtures, baseline comparisons, and executable pseudocode.
- `SymPy` is always present, even if the note says symbolic treatment is mostly ceremonial for that paper.
- `tinygrad` is always present, even if the note says the backend is mostly ceremonial for that paper.
- `tinygrad` is the first minimal executable autodiff backend.
- This repo defaults tinygrad to the `LLVM` backend inside the tinygrad-specific code paths when no tinygrad backend env var is already set.
- `PyTorch` is the primary executable training implementation.
- `JAX` is the second executable implementation and a cross-check on functional structure.
- `Cubical Agda` is always present, even if the note says the formalization is intentionally thin.

Agda status note:
- `make agda-check` means the Cubical Agda layer typechecks successfully.
- It does not mean the whole paper is formally verified.
- In many papers here, the Agda layer is intentionally minimal and captures only a formal core, interface, or invariant slice.

## Why This Pipeline

This repository is not trying to collect random backend ports. The point of the stack is that each layer answers a different question about the same paper.

### The Short Version

```text
NumPy -> SymPy -> tinygrad -> PyTorch -> JAX -> Cubical Agda
```

Read it like this:

- `NumPy`: what is the numerical object?
- `SymPy`: what is the symbolic formula?
- `tinygrad`: what is the smallest real autodiff implementation of that formula?
- `PyTorch`: what is the practical, production-grade implementation?
- `JAX`: what does the same system look like in a functional transformation-oriented style?
- `Cubical Agda`: what can be stated and checked at the level of types, invariants, and proofs?

That ordering is deliberate. It moves from direct manipulation, to derivation, to minimal autodiff, to industrial tooling, to functional cross-checking, to formal structure.

### What Each Layer Contributes

`NumPy` is the foundation because it leaves very little hidden.

- Arrays, linear algebra, and eager numerical execution are explicit.
- There is no automatic differentiation engine to hide mistakes.
- If a paper uses a recurrence, an attention score, a KL term, or a convolution, a NumPy implementation forces the repository to say exactly what that object is in ordinary numerical terms.

`SymPy` sits directly above that because it answers a different question from NumPy.

- NumPy tells you the value for an input.
- SymPy tells you the formula, the derivative, the simplification, or the identity behind that value.
- This is the layer where the repo can justify gradient formulas, ELBO algebra, gate equations, receptive-field arithmetic, and attention-score derivations without hiding behind code alone.

`tinygrad` comes next because it is the first genuinely executable autodiff backend that is still small enough to feel transparent.

- It lets the repo move from hand-written math to a real framework tensor/autograd model.
- Unlike larger frameworks, the implementation remains compact enough that the backend still serves the educational goal of the repo rather than overwhelming it.
- In this project, `tinygrad` is the bridge between “I can derive this” and “I can run this in a framework without losing the conceptual thread.”

`PyTorch` remains after `tinygrad`, not before it.

- PyTorch is the main practical reference backend in the repo.
- It is the place where the implementation should be easiest to extend, train, debug, and compare against common practice.
- It has the richest ergonomics for most papers here, but that is exactly why it should not be the first executable layer. By the time code reaches PyTorch in this pipeline, the repo should already know what it is trying to say.

`JAX` comes after PyTorch because the project uses it as a second serious executable interpretation, not as the canonical first one.

- JAX forces clearer parameter/state separation.
- JAX makes the function transformation view explicit: `grad`, `jit`, `vmap`, and related structure.
- It is valuable as a parity backend because agreement between PyTorch and JAX catches a class of implementation drift that a single-framework repo would miss.

`Cubical Agda` comes last because it is not “just another backend.”

- It is the layer for signatures, invariants, structural interfaces, and proofs.
- For some papers this means a meaningful formal core.
- For others it means a deliberately thin but explicit statement of what is worth formalizing and what would be ceremonial.
- The repo keeps it present across all papers because completeness matters, but it does not pretend that every paper deserves the same proof effort.

### Why This Order Is Better Than Random Backend Accumulation

The important point is that the layers are not interchangeable.

- `SymPy` is not a weaker `tinygrad`.
- `tinygrad` is not a smaller `PyTorch`.
- `JAX` is not just “PyTorch but different syntax.”
- `Cubical Agda` is not an implementation backend in the ordinary sense at all.

They do different jobs:

- `NumPy` gives direct executable mathematics.
- `SymPy` gives algebraic explanation.
- `tinygrad` gives minimal autodiff execution.
- `PyTorch` gives practical implementation depth.
- `JAX` gives functional cross-verification.
- `Cubical Agda` gives formal structure.

This is why the repo uses one canonical pipeline instead of treating the backends as a flat checklist.

### Why tinygrad Belongs Here

Adding `tinygrad` improves the stack because there was previously a gap between symbolic derivation and industrial frameworks.

Without `tinygrad`, the jump looked like this:

- symbolic formulas in `SymPy`
- then immediately into `PyTorch` and `JAX`

That works, but it skips an important explanatory layer. `tinygrad` fills that gap by being:

- executable
- differentiable
- framework-shaped
- still small enough to remain legible

That makes it especially useful for:

- RNNs
- LSTMs
- small attention mechanisms
- compact CNNs
- VAEs and other papers where the core tensor program matters more than ecosystem integrations

It is less informative for some papers with a heavier systems or evaluation emphasis, and the repo records that explicitly in paper notes. But even there, the policy is the same as with `SymPy` and `Cubical Agda`: keep the layer present, and be honest when it is thin.

### What the Repository Gains From the Full Stack

By the end of this pipeline, a paper in the repo can be understood at multiple levels:

- as direct numerical code
- as symbolic mathematics
- as a minimal autodiff program
- as a practical training implementation
- as a functional parity implementation
- as a formal object with explicit invariants

That is the real purpose of the project. It is not only to “have many implementations.” It is to make each paper legible from calculation, to derivation, to execution, to verification.

### The Practical Rule

For implemented papers, the default expectation is:

- `NumPy`: tiny checks only
- `SymPy`: always present
- `tinygrad`: always present
- `PyTorch`: substantive
- `JAX`: substantive
- `Cubical Agda`: always present

And when a layer is low-value for a paper, the repository should say so plainly in `NOTES.md` rather than faking depth.

## Why This Repo Matters

This repository is most useful as a research-training and research-clarification project.

It is not primarily trying to be:

- a leaderboard repo
- a production benchmark suite
- a claim that every paper here has been reproduced at full original scale

It is trying to do something narrower and, for many researchers, more durable:

- make important ML papers executable in small form
- make their mathematics explicit rather than merely implied
- make backend agreement part of the method
- make notes about thin or ceremonial layers explicit instead of pretending every layer contributes equally

### Who This Is For

This repo is especially useful for:

- early-stage AI/ML researchers who want to move from framework fluency to first-principles understanding
- research engineers who want parity checks across multiple backend styles
- theory-minded ML readers who care about the distinction between empirical behavior, symbolic derivation, and formal structure
- teachers and self-learners who want a paper to exist as more than one code artifact

It is less useful for:

- readers who only want the fastest production implementation
- researchers whose only criterion is original-scale benchmark reproduction
- people looking for a single-framework “best practices” repo

### The Core Usefulness

The main value is triangulation.

A paper in this repository is not reduced to one implementation language and one style of correctness. Instead, it is seen through several different lenses:

- `NumPy`: the smallest direct numerical statement
- `SymPy`: the algebraic or derivational statement
- `tinygrad`: the smallest real autodiff framework statement
- `PyTorch`: the practical and extensible statement
- `JAX`: the functional parity statement
- `Cubical Agda`: the typed and formal statement

That means the repository can help answer different kinds of questions:

- What is this paper actually computing?
- What equations justify that computation?
- What does autodiff have to recover?
- Does the implementation survive translation across backend paradigms?
- What structural invariant is worth stating explicitly?

For current researchers, that is useful as a debugging and understanding discipline.

For future researchers, it can become a reference corpus for how to study an ML idea across multiple representational layers instead of treating “the PyTorch version” as the whole object.

### Related Work And What Is Unusual Here

This project is not the first educational implementation effort, and it is not the first multi-framework effort.

There are clear neighboring precedents:

- *Dive into Deep Learning* shows that educational material can be written across multiple frameworks.
- *The Annotated Transformer* is a classic example of deeply explanatory paper-to-code exposition.
- framework-bridging projects such as `EagerPy` and multi-backend scientific ML libraries show that common logic can span several array/tensor systems.
- `tinygrad` itself demonstrates the value of a small, inspectable autodiff framework.
- proof-assistant work around neural-network-adjacent mathematics shows that formal methods can be brought into ML-adjacent domains.

What seems unusual here is the synthesis.

This repository deliberately combines all of the following:

- a fixed paper corpus
- a dependency-aware build order
- a standing multi-layer pipeline
- always-present symbolic and formal layers, even when thin
- backend parity as a normal expectation rather than an optional extra
- paper notes that explicitly say when a layer is low-value or ceremonial

So the claim is not “nothing like this has ever existed.”

The stronger and more defensible claim is:

- the components all have precedents
- the combination is unusual
- the method is the point

### The Research Arc So Far

The repository has already taken on a recognizable shape.

The early build order established the grammar of deep learning:

- recurrence
- gating
- regularization
- convolution
- residual learning

The middle of the build order moved into structure and mechanism:

- attention
- pointer-style outputs
- set encoding
- graph message passing
- memory systems
- latent-variable models

That matters because later papers no longer need to invent their own local language from scratch. They inherit a vocabulary of tested objects, shapes, losses, and invariants from earlier papers.

### The Methodological Arc

The process only works if every paper is reduced to a disciplined small core.

The repo’s method is:

1. write a precise `spec.md`
2. choose a tiny deterministic problem
3. make the NumPy layer the smallest executable truth source
4. add symbolic, minimal autodiff, practical, and functional backends
5. keep the formal layer present even when thin
6. test parity and one-step behavior instead of just checking that files exist
7. state omissions honestly in `NOTES.md`

That is the real research method embedded in the repo. It is less about collecting code and more about building stable comparative understanding.

## Repository Layout

```text
sutskever-30-beyond-numpy/
├── README.md
├── papers.yaml
├── shared/
│   ├── fixtures/
│   ├── tests/
│   └── notes/
├── templates/
│   └── paper-template.md
└── papers/
    ├── 01_complexity_dynamics/
    ├── 02_char_rnn_karpathy/
    ├── ...
    └── 30_lost_in_middle/
```

Each paper directory contains:

- `README.md`
- `spec.md`
- `NOTES.md`
- `numpy_checks.py`
- `sympy/`
- `tinygrad/`
- `torch/`
- `jax/`
- `cubical-agda/`
- `tests/`

## Canonical Order vs Build Order

Canonical order remains `01..30`.

Recommended build order:

1. `26` CS231n
2. `02` Char RNN
3. `03` LSTM
4. `04` RNN Regularization
5. `07` AlexNet
6. `10` ResNet
7. `15` Identity Mappings in ResNet
8. `11` Dilated Convolutions
9. `14` Bahdanau Attention
10. `06` Pointer Networks
11. `08` Seq2Seq for Sets
12. `13` Attention Is All You Need
13. `16` Relational Reasoning
14. `18` Relational RNN
15. `20` Neural Turing Machine
16. `12` Neural Message Passing
17. `21` Deep Speech 2 / CTC
18. `17` Variational Lossy Autoencoder
19. `09` GPipe
20. `22` Scaling Laws
21. `27` Multi-token Prediction
22. `28` Dense Passage Retrieval
23. `29` Retrieval-Augmented Generation
24. `30` Lost in the Middle
25. `05` Keeping Neural Networks Simple
26. `23` MDL Principle
27. `25` Kolmogorov Complexity
28. `24` Machine Super Intelligence
29. `01` First Law of Complexodynamics
30. `19` Coffee Automaton

The structured source of truth for this is [`papers.yaml`](/Users/hifi/sutskever-30-beyond-numpy/papers.yaml).

## Commands

```bash
source scripts/env.sh
python3 -m pytest papers/02_char_rnn_karpathy/tests/test_char_rnn.py papers/03_lstm_understanding/tests/test_lstm.py papers/26_cs231n_cnn_fundamentals/tests/test_cnn.py -q
python3 scripts/run_paper.py --paper 02
python3 scripts/run_paper.py --paper 03
python3 scripts/run_paper.py --paper 26
make agda-check
```

If `agda` is not already on your shell `PATH`, run:

```bash
source scripts/env.sh
```

To make that persistent in `zsh`, add this line to `~/.zshrc`:

```bash
export PATH="/Users/hifi/Library/Python/3.9/bin:$PATH"
```

## tinygrad Install

For this repo, use:

```bash
./scripts/install_tinygrad.sh
```

That installs `tinygrad` without the failing optional macOS Metal dependency chain. The repo's tinygrad code then defaults to `LLVM` unless you explicitly choose another tinygrad backend.

## License

Educational use under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. See individual papers for original research citations.

## Citation

If you use these implementations in your work or teaching:

```bibtex
@misc{sutskever30beyondnumpy,
  title={Sutskever 30 Beyond NumPy: Multi-Backend Educational Implementation Suite},
  author={Paul "The Pageman" Pajo and collaborators},
  year={2026},
  note={Educational multi-backend implementations of papers from Ilya Sutskever's recommended reading list, based on the NumPy-version-only repository https://github.com/pageman/Sutskever-30-Implementations}
}
```
