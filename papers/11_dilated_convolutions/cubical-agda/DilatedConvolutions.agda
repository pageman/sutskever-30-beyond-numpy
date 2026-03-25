{-# OPTIONS --cubical #-}

module DilatedConvolutions where

open import Agda.Builtin.Equality

postulate
  Image Feature Label : Set

record DilatedLayer : Set1 where
  field
    apply : Image -> Feature

open DilatedLayer public

record Classifier : Set1 where
  field
    classify : Feature -> Label

open Classifier public

forward : DilatedLayer -> Classifier -> Image -> Label
forward layer head x = classify head (apply layer x)

forward-unfold : (layer : DilatedLayer) (head : Classifier) (x : Image) -> forward layer head x ≡ classify head (apply layer x)
forward-unfold layer head x = refl
