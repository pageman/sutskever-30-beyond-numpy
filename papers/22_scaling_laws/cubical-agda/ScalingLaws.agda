{-# OPTIONS --cubical #-}

module ScalingLaws where

open import Agda.Builtin.Equality

postulate
  Size Loss Prediction : Set

record Fit : Set1 where
  field
    predict : Size -> Prediction

open Fit public

record Observe : Set1 where
  field
    observe : Prediction -> Loss

open Observe public

forward : Fit -> Observe -> Size -> Loss
forward f o n = observe o (predict f n)

forward-unfold : (f : Fit) (o : Observe) (n : Size) -> forward f o n ≡ observe o (predict f n)
forward-unfold f o n = refl
