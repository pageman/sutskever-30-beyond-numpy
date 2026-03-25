{-# OPTIONS --cubical #-}

module MachineSuperIntelligence where

open import Agda.Builtin.Equality

postulate
  Capability Consequence : Set

record Aggregator : Set1 where
  field
    assess : Capability -> Consequence

open Aggregator public

assess-self : (a : Aggregator) (c : Capability) -> assess a c ≡ assess a c
assess-self a c = refl
