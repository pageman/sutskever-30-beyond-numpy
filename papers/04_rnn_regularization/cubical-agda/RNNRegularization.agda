{-# OPTIONS --cubical #-}

module RNNRegularization where

open import Agda.Builtin.Equality

postulate
  Token Hidden : Set

record Regularizer : Set1 where
  field
    applyInputMask : Token -> Token
    applyHiddenMask : Hidden -> Hidden

open Regularizer public

record RNNParams : Set1 where
  field
    stepHidden : Token -> Hidden -> Hidden

open RNNParams public

step : Regularizer -> RNNParams -> Token -> Hidden -> Hidden
step reg params x h = stepHidden params (applyInputMask reg x) (applyHiddenMask reg h)

step-unfold : (reg : Regularizer) (params : RNNParams) (x : Token) (h : Hidden) -> step reg params x h ≡ stepHidden params (applyInputMask reg x) (applyHiddenMask reg h)
step-unfold reg params x h = refl
