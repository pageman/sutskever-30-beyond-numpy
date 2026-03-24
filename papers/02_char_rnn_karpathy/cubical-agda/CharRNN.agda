{-# OPTIONS --cubical #-}

module CharRNN where

record RNNShape : Set1 where
  field
    V H : Set

open RNNShape public

record RNNParams (S : RNNShape) : Set1 where
  field
    embed : V S -> H S
    recurrent : H S -> H S
    project : H S -> V S

open RNNParams public

record RNNState (S : RNNShape) : Set where
  field
    hidden : H S

open RNNState public

step :
  {S : RNNShape} ->
  RNNParams S ->
  V S ->
  RNNState S ->
  RNNState S
step params x state = record { hidden = recurrent params (hidden state) }
