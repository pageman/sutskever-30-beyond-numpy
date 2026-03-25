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

cong : {A B : Set} {x y : A} -> (f : A -> B) -> x ≡ y -> f x ≡ f y
cong f refl = refl

step-cong-memory
  : (addr : Addressing)
  -> (c : Controller)
  -> {m₁ m₂ : Memory}
  -> m₁ ≡ m₂
  -> step addr c m₁ ≡ step addr c m₂
step-cong-memory addr c m≡ = cong (write addr c) m≡
