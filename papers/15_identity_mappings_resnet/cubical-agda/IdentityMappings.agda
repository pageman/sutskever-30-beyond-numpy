{-# OPTIONS --cubical #-}

module IdentityMappings where

open import Agda.Builtin.Equality

postulate
  Feature Label : Set

record PreActBlock : Set1 where
  field
    residual : Feature -> Feature

open PreActBlock public

identity : Feature -> Feature
identity x = x

identity-refl : (x : Feature) -> identity x ≡ x
identity-refl x = refl

blockForward : PreActBlock -> Feature -> Feature
blockForward block x = identity x

record Classifier : Set1 where
  field
    classify : Feature -> Label

open Classifier public

forward : PreActBlock -> Classifier -> Feature -> Label
forward block head x = classify head (blockForward block x)
