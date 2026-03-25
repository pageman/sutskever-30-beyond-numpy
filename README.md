# Sutskever 30 Beyond NumPy

Multi-backend implementations of the 30 papers from the Sutskever reading list, using the canonical NumPy repo as the baseline reference and specification.

Reference baseline:
- [`pageman/sutskever-30-implementations`](https://github.com/pageman/sutskever-30-implementations)

This repository keeps the original paper numbering for compatibility, but tracks a separate `build_order` so implementation work can follow dependencies rather than the original list order.

## Tooling

- Python project metadata: [`pyproject.toml`](/Users/hifi/sutskever-30-beyond-numpy/pyproject.toml)
- Demo runner: [`scripts/run_paper.py`](/Users/hifi/sutskever-30-beyond-numpy/scripts/run_paper.py)
- Agda library file: [`sutskever-30-beyond-numpy.agda-lib`](/Users/hifi/sutskever-30-beyond-numpy/sutskever-30-beyond-numpy.agda-lib)
- Agda setup notes: [`docs/AGDA_SETUP.md`](/Users/hifi/sutskever-30-beyond-numpy/docs/AGDA_SETUP.md)

## Backend Policy

Per paper, the expected implementation pipeline is:

1. `spec.md`
2. `numpy_checks.py`
3. `sympy/`
4. `torch/`
5. `jax/`
6. `cubical-agda/`

Rules:
- `NumPy` is minimal and exists only for sanity checks, fixtures, baseline comparisons, and executable pseudocode.
- `SymPy` is always present, even if the note says symbolic treatment is mostly ceremonial for that paper.
- `PyTorch` is the primary executable training implementation.
- `JAX` is the second executable implementation and a cross-check on functional structure.
- `Cubical Agda` is always present, even if the note says the formalization is intentionally thin.

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
python3 -m pytest papers/02_char_rnn_karpathy/tests/test_char_rnn.py papers/03_lstm_understanding/tests/test_lstm.py papers/26_cs231n_cnn_fundamentals/tests/test_cnn.py -q
python3 scripts/run_paper.py --paper 02
python3 scripts/run_paper.py --paper 03
python3 scripts/run_paper.py --paper 26
make agda-check
```
