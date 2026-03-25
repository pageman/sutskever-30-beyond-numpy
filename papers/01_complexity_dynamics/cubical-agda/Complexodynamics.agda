{-# OPTIONS --cubical #-}

module Complexodynamics where

open import Agda.Builtin.Equality

postulate
  State Structure : Set

record Flow : Set1 where
  field
    evolve : State -> Structure

open Flow public

evolve-self : (f : Flow) (s : State) -> evolve f s ≡ evolve f s
evolve-self f s = refl
