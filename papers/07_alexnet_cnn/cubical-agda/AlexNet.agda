{-# OPTIONS --cubical #-}

module AlexNet where

open import Agda.Builtin.Equality

postulate
  Image Feature HiddenFeature Label : Set

record FeatureExtractor : Set1 where
  field
    conv₁ : Image -> Feature
    conv₂ : Feature -> HiddenFeature

open FeatureExtractor public

record Classifier : Set1 where
  field
    classify : HiddenFeature -> Label

open Classifier public

forward : FeatureExtractor -> Classifier -> Image -> Label
forward extractor head x = classify head (conv₂ extractor (conv₁ extractor x))

forward-unfold : (extractor : FeatureExtractor) (head : Classifier) (x : Image) -> forward extractor head x ≡ classify head (conv₂ extractor (conv₁ extractor x))
forward-unfold extractor head x = refl
