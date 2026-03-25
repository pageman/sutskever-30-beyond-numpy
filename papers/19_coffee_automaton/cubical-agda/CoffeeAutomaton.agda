{-# OPTIONS --cubical #-}

module CoffeeAutomaton where

open import Agda.Builtin.Equality

postulate
  State Output : Set

record Automaton : Set1 where
  field
    step : State -> Output

open Automaton public

step-self : (a : Automaton) (s : State) -> step a s ≡ step a s
step-self a s = refl
