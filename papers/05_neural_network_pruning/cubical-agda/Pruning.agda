{-# OPTIONS --cubical #-}

module Pruning where

open import Agda.Builtin.Equality

postulate
  Input Hidden Output : Set

record SparseNetwork : Set1 where
  field
    keep  : Input -> Hidden
    pruned : Input -> Hidden
    readout : Hidden -> Output

open SparseNetwork public

keep-self : (net : SparseNetwork) (x : Input) -> keep net x ≡ keep net x
keep-self net x = refl
