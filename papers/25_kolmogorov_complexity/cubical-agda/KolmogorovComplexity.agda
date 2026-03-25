{-# OPTIONS --cubical #-}

module KolmogorovComplexity where

open import Agda.Builtin.Equality

postulate
  Sequence Description Complexity : Set

record Descriptor : Set1 where
  field
    describe : Sequence -> Description
    measure  : Description -> Complexity

open Descriptor public

measure-self : (d : Descriptor) (s : Sequence) -> measure d (describe d s) ≡ measure d (describe d s)
measure-self d s = refl
