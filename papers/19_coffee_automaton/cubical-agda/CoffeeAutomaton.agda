{-# OPTIONS --cubical #-}

module CoffeeAutomaton where

postulate
  State Output : Set

record Automaton : Set1 where
  field
    step : State -> Output

open Automaton public
