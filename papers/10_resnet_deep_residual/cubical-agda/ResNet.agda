{-# OPTIONS --cubical #-}

module ResNet where

open import Agda.Builtin.Equality

postulate
  Feature Label : Set

record ResidualBlock : Set1 where
  field
    residual : Feature -> Feature

open ResidualBlock public

record Classifier : Set1 where
  field
    classify : Feature -> Label

open Classifier public

blockForward : ResidualBlock -> Feature -> Feature
blockForward block x = x

blockForward-identity : (block : ResidualBlock) (x : Feature) -> blockForward block x ≡ x
blockForward-identity block x = refl

forward : ResidualBlock -> Classifier -> Feature -> Label
forward block head x = classify head (blockForward block x)
