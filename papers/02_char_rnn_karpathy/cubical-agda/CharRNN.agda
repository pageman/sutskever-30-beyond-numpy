{-# OPTIONS --cubical #-}

module CharRNN where

postulate
  V H : Set

record RNNShape : Set1 where
  field
    vocab hidden : Set

record RNNParams : Set1 where
  field
    embed : V -> H
    recurrent : H -> H
    project : H -> V

open RNNParams public

record RNNState : Set where
  field
    state : H

open RNNState public

step : RNNParams -> V -> RNNState -> RNNState
step params x current = record { state = recurrent params (state current) }
