{-# OPTIONS --cubical #-}

module IdentityMappings where

postulate
  Feature Label : Set

record PreActBlock : Set1 where
  field
    residual : Feature -> Feature

open PreActBlock public

identity : Feature -> Feature
identity x = x

blockForward : PreActBlock -> Feature -> Feature
blockForward block x = identity x

record Classifier : Set1 where
  field
    classify : Feature -> Label

open Classifier public

forward : PreActBlock -> Classifier -> Feature -> Label
forward block head x = classify head (blockForward block x)
