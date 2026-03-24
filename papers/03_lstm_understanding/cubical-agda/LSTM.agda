{-# OPTIONS --cubical #-}

module LSTM where

record LSTMShape : Set1 where
  field
    V H : Set

open LSTMShape public

record LSTMState (S : LSTMShape) : Set where
  field
    hidden : H S
    cell : H S

open LSTMState public

record Gates (S : LSTMShape) : Set where
  field
    forget : H S
    input : H S
    output : H S
    candidate : H S

open Gates public

step :
  {S : LSTMShape} ->
  Gates S ->
  LSTMState S ->
  LSTMState S
step gates state = record { hidden = output gates ; cell = candidate gates }
