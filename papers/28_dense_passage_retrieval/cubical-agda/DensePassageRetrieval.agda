{-# OPTIONS --cubical #-}

module DensePassageRetrieval where

open import Agda.Builtin.Equality

postulate
  Query Passage Score : Set

record Encoder : Set1 where
  field
    encodeQ : Query -> Score
    encodeP : Passage -> Score

open Encoder public

encodeQ-self : (e : Encoder) (q : Query) -> encodeQ e q ≡ encodeQ e q
encodeQ-self e q = refl
