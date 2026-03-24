{-# OPTIONS --cubical #-}

module CNN where

record ImageShape : Set1 where
  field
    Pixel Feature Label : Set

open ImageShape public

record ConvLayer (S : ImageShape) : Set1 where
  field
    convolve : Pixel S -> Feature S

open ConvLayer public

record Classifier (S : ImageShape) : Set1 where
  field
    classify : Feature S -> Label S

open Classifier public

forward :
  {S : ImageShape} ->
  ConvLayer S ->
  Classifier S ->
  Pixel S ->
  Label S
forward conv head x = classify head (convolve conv x)
