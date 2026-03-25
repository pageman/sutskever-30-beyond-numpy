{-# OPTIONS --cubical #-}

module MultiTokenPrediction where

open import Agda.Builtin.Equality

postulate
  Context Token₁ Token₂ : Set

record Predictor : Set1 where
  field
    predict₁ : Context -> Token₁
    predict₂ : Context -> Token₂

open Predictor public

predict₁-self : (p : Predictor) (c : Context) -> predict₁ p c ≡ predict₁ p c
predict₁-self p c = refl
