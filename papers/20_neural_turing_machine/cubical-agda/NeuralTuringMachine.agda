{-# OPTIONS --cubical #-}

module NeuralTuringMachine where

open import Agda.Builtin.Equality

postulate
  Controller Memory Output : Set

record Addressing : Set1 where
  field
    read : Controller -> Memory -> Output
    write : Controller -> Memory -> Memory

open Addressing public

step : Addressing -> Controller -> Memory -> Memory
step addr c m = write addr c m

step-unfold : (addr : Addressing) (c : Controller) (m : Memory) -> step addr c m ≡ write addr c m
step-unfold addr c m = refl
