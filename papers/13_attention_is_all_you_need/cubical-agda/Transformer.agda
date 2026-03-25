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

cong : {A B : Set} {x y : A} -> (f : A -> B) -> x ≡ y -> f x ≡ f y
cong f refl = refl

forward-cong-context
  : (a : Attention)
  -> (r : Readout)
  -> (q₁ q₂ k₁ k₂ : Token)
  -> attend a q₁ k₁ ≡ attend a q₂ k₂
  -> forward a r q₁ k₁ ≡ forward a r q₂ k₂
forward-cong-context a r q₁ q₂ k₁ k₂ attn≡ = cong (project r) attn≡
