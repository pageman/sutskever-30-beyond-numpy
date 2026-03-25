{-# OPTIONS --cubical #-}

module CTC where

open import Agda.Builtin.Equality

postulate
  Alignment Symbol Sequence : Set

record Collapse : Set1 where
  field
    collapse : Alignment -> Sequence

open Collapse public

record Decoder : Set1 where
  field
    emit : Sequence -> Symbol

open Decoder public

forward : Collapse -> Decoder -> Alignment -> Symbol
forward ctc dec a = emit dec (collapse ctc a)

forward-unfold : (ctc : Collapse) (dec : Decoder) (a : Alignment) -> forward ctc dec a ≡ emit dec (collapse ctc a)
forward-unfold ctc dec a = refl

cong : {A B : Set} {x y : A} -> (f : A -> B) -> x ≡ y -> f x ≡ f y
cong f refl = refl

forward-cong-collapse
  : (ctc : Collapse)
  -> (dec : Decoder)
  -> (a₁ a₂ : Alignment)
  -> collapse ctc a₁ ≡ collapse ctc a₂
  -> forward ctc dec a₁ ≡ forward ctc dec a₂
forward-cong-collapse ctc dec a₁ a₂ collapse≡ = cong (emit dec) collapse≡
