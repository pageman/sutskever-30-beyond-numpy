{-# OPTIONS --cubical #-}

module Transformer where

open import Agda.Builtin.Equality

postulate
  Token Context Output : Set

record Attention : Set1 where
  field
    attend : Token -> Token -> Context

open Attention public

record Readout : Set1 where
  field
    project : Context -> Output

open Readout public

forward : Attention -> Readout -> Token -> Token -> Output
forward a r q k = project r (attend a q k)

forward-unfold : (a : Attention) (r : Readout) (q k : Token) -> forward a r q k ≡ project r (attend a q k)
forward-unfold a r q k = refl
