# Agda Setup

This repository uses Cubical Agda for the formalization layer.

## What You Need

- `agda`
- `ghc`
- `cabal` or another Agda installation route

For the current first populated papers (`26`, `02`, `03`), the Agda files are intentionally self-contained and do not depend on the Agda standard library. That keeps the first typechecking path as small as possible.

## Fastest Practical Install On macOS

If Homebrew is available:

```bash
brew install agda
```

Then verify:

```bash
source scripts/env.sh
agda --version
make agda-check
```

## Alternative Haskell-First Route

1. Install `ghc` and `cabal`
2. Install `Agda`
3. Add `agda` to your `PATH`

## Repo Files

- `sutskever-30-beyond-numpy.agda-lib`
- `Makefile`
- `papers/02_char_rnn_karpathy/cubical-agda/CharRNN.agda`
- `papers/03_lstm_understanding/cubical-agda/LSTM.agda`
- `papers/26_cs231n_cnn_fundamentals/cubical-agda/CNN.agda`

## Notes

- Cubical mode is enabled per file with `{-# OPTIONS --cubical #-}`.
- These first Agda modules are structural, not numeric.
- Later papers can opt into the standard library if the formalization actually benefits from it.
- If Agda was installed into `/Users/hifi/Library/Python/3.9/bin`, source `scripts/env.sh` or add that directory to your shell startup file.
