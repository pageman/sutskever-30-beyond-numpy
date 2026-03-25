{-# OPTIONS --cubical #-}

module CNN where

open import Agda.Builtin.Equality

postulate
  Pixel Feature Label : Set

record ConvLayer : Set1 where
  field
    convolve : Pixel -> Feature

open ConvLayer public

record Classifier : Set1 where
  field
    classify : Feature -> Label

open Classifier public

forward : ConvLayer -> Classifier -> Pixel -> Label
forward conv head x = classify head (convolve conv x)

forward-unfold : (conv : ConvLayer) (head : Classifier) (x : Pixel) -> forward conv head x ≡ classify head (convolve conv x)
forward-unfold conv head x = refl
