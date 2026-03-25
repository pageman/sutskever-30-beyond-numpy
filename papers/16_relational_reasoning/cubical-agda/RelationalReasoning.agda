{-# OPTIONS --cubical #-}

module RelationalReasoning where

postulate
  Object RelationFeature Label : Set

record RelationNet : Set1 where
  field
    relate : Object -> Object -> RelationFeature

open RelationNet public

record Classifier : Set1 where
  field
    classify : RelationFeature -> Label

open Classifier public

forward : RelationNet -> Classifier -> Object -> Object -> Label
forward net head a b = classify head (relate net a b)
