{-# OPTIONS --cubical #-}

module RelationalRNN where

open import Agda.Builtin.Equality

postulate
  Slot Output : Set

record RelationMemory : Set1 where
  field
    relate : Slot -> Slot -> Slot
    update : Slot -> Slot

open RelationMemory public

step : RelationMemory -> Slot -> Slot
step mem s = update mem (relate mem s s)

step-unfold : (mem : RelationMemory) (s : Slot) -> step mem s ≡ update mem (relate mem s s)
step-unfold mem s = refl
