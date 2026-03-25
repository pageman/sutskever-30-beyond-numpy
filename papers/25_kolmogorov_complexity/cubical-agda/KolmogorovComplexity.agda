{-# OPTIONS --cubical #-}

module KolmogorovComplexity where

postulate
  Sequence Description Complexity : Set

record Descriptor : Set1 where
  field
    describe : Sequence -> Description
    measure  : Description -> Complexity

open Descriptor public
