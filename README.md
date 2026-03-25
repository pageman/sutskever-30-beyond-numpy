# Sutskever 30 Beyond NumPy

Multi-backend implementations of the 30 papers from the Sutskever reading list, using the canonical NumPy repo as the baseline reference and specification.

Reference baseline:
- [`pageman/sutskever-30-implementations`](https://github.com/pageman/sutskever-30-implementations)

This repository keeps the original paper numbering for compatibility, but tracks a separate `build_order` so implementation work can follow dependencies rather than the original list order.

## About

Based on the Numpy-version-only:
- https://github.com/pageman/sutskever-30-implementations

## Tooling

- Python project metadata: [`pyproject.toml`](pyproject.toml)
- Demo runner: [`scripts/run_paper.py`](scripts/run_paper.py)
- Verification telemetry generator: [`scripts/generate_verification_status.py`](scripts/generate_verification_status.py)
- tinygrad installer helper: [`scripts/install_tinygrad.sh`](scripts/install_tinygrad.sh)
- Agda library file: [`sutskever-30-beyond-numpy.agda-lib`](sutskever-30-beyond-numpy.agda-lib)
- Agda setup notes: [`docs/AGDA_SETUP.md`](docs/AGDA_SETUP.md)

Reserved shared space:
- [`shared/fixtures`](shared/fixtures)
- [`shared/tests`](shared/tests)
- [`shared/notes`](shared/notes)

Those directories are future-facing reserved space for consolidation. The active repo design today keeps fixtures, notes, and tests paper-local unless repeated patterns become worth extracting.

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

Verification telemetry note:
- this repo exposes a generated [`verification.yaml`](verification.yaml) as an observability artifact, not as a correctness certificate
- the point is to record what was present, executed, typechecked, and last refreshed
- the point is not to claim full reproduction or full formal verification

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

### Narrative Arc

The finished build order reads like a research apprenticeship arc.

It starts by teaching the grammar of learning systems:

- `26`, `02`, `03`, `04`
- direct classifiers, vanilla recurrence, gated recurrence, and regularization

It then builds the main architectural backbone:

- `07`, `10`, `15`, `11`
- convolution, depth, residual routing, identity mappings, and receptive-field control

It then shifts from raw architecture into structured computation:

- `14`, `06`, `08`, `13`, `16`, `18`, `20`, `12`, `21`, `17`
- alignment, pointing, set structure, attention, relational reasoning, memory, graphs, sequence alignment, and latent variables

It then arrives at modern large-model and retrieval behavior:

- `09`, `22`, `27`, `28`, `29`, `30`
- systems scaling, scaling-law abstractions, multi-token prediction, retrieval, retrieval-augmented generation, and context-position effects

It closes with a deliberately theory-heavy tail:

- `05`, `23`, `25`, `24`, `01`, `19`
- simplicity pressure, description length, compressibility, capability aggregation, toy complexity dynamics, and automaton structure

The result is cumulative language. Later papers do not arrive in isolation. They inherit notation, test discipline, backend expectations, parity habits, and formalization norms from earlier ones. That matters because the educational value of a corpus is not just in having many papers; it is in making the later ones easier to think with because the earlier ones already established the conceptual grammar.

> **Narrative / Story Box**
> The story of the repo is a climb from hand-checkable mechanisms to systems-level behavior and then into reflective, theory-adjacent territory. It begins with learning how to state and train simple objects correctly, moves through memory, attention, and retrieval, and ends in a tail where the notes, claims, and formal slices are often as important as the executable toy itself.

### Methodological Arc

The methodological arc matters as much as the paper sequence because the repo is not only collecting implementations. It is repeatedly forcing the same reduction:

- what is the smallest executable object here?
- what is the core equation or invariant?
- what must survive translation across backends?
- what is intentionally thin, partial, or ceremonial?

In the early papers, the difficulty is mostly mechanical:

- state the object clearly
- make the tensor program tiny and deterministic
- force agreement across backends

In the middle papers, the difficulty becomes architectural:

- preserve shape discipline
- preserve attention or memory semantics
- keep parity without hiding complexity behind framework magic

In the later papers, the difficulty becomes interpretive:

- decide which part is executable mechanism and which part is conceptual framing
- decide what `SymPy` can illuminate versus what it can only restate
- decide what `Cubical Agda` can formalize honestly versus what would be fake rigor

That shift is deliberate. Early papers train implementation discipline. Later papers train representational discipline.

The recurring method of the repo is:

1. write a precise `spec.md`
2. choose a tiny deterministic problem or toy object
3. make `NumPy` the smallest executable truth source
4. add `SymPy`, `tinygrad`, `PyTorch`, and `JAX` as distinct interpretive and executable layers
5. keep `Cubical Agda` present as a real formal layer even when it is minimal
6. test parity, smoke behavior, and runner coverage instead of only checking file presence
7. state omissions and low-value layers plainly in `NOTES.md`

The end result is a comparative method, not just a code archive. Each paper becomes a stable object that can be inspected numerically, symbolically, through a small autodiff system, through mainstream frameworks, through a functional transformation stack, and through a typed formal layer.

> **Method Box**
> The method of the repo is triangulation under constraint: one paper, one compact spec, one tiny deterministic object, one minimal numerical truth source, several executable translations, one honest note about what is thin, and one formal layer that is real even when small.

> **Narrative / Story Box**
> If the narrative arc is “the ideas get larger,” the methodological arc is “the standards get stricter.” The repo begins by teaching how to implement the math plainly, and it ends by demanding honesty about what is actually executable, what is merely derivational, and what is formalized in a meaningful way.

> **Method Box**
> In practice, the repo works best when every paper answers four questions clearly: what is the smallest executable object, what is the core equation, what do the backends have to agree on, and what is intentionally omitted.

## Status

The full `01..30` corpus is now populated.

Current verification status:

- `61` Python tests passing
- `make agda-check` passing across all paper formalization layers
- `scripts/run_paper.py` wired for all `30` papers

That does not mean all papers are implemented at equal depth. It does mean the repository is structurally complete and verified at the level this project claims.

## Verification Telemetry

This repository keeps a generated [`verification.yaml`](verification.yaml) file as a repo-level verification record.

It is meant to answer:

- which layers are present for each paper
- which repo-wide checks most recently passed
- whether the Agda layer typechecked
- whether the demo runner sweep completed
- which layers are substantive, partial, minimal, or ceremonial

It is intentionally not framed as a certificate. The right interpretation is observability and refreshable status, not authority.

Refresh it with:

```bash
python3 scripts/generate_verification_status.py --run-checks
```

or:

```bash
make verification-status
```

## Repository Layout

```text
sutskever-30-beyond-numpy/
├── README.md
├── papers.yaml
├── docs/
├── shared/                    # Reserved for future shared fixtures/tests/notes
├── scripts/
├── src/
│   └── s30bn/
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

The structured source of truth for this is [`papers.yaml`](papers.yaml).

## Commands

```bash
source scripts/env.sh
make test
python3 scripts/run_paper.py --paper 01
python3 scripts/run_paper.py --paper 19
python3 scripts/run_paper.py --paper 30
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
