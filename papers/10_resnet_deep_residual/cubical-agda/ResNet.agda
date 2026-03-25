{-# OPTIONS --cubical #-}

module ResNet where

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

forward : ResidualBlock -> Classifier -> Feature -> Label
forward block head x = classify head (blockForward block x)
