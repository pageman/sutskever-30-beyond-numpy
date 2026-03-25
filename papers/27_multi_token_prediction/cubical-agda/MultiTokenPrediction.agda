{-# OPTIONS --cubical #-}

module MultiTokenPrediction where

postulate
  Context Token₁ Token₂ : Set

record Predictor : Set1 where
  field
    predict₁ : Context -> Token₁
    predict₂ : Context -> Token₂

open Predictor public
