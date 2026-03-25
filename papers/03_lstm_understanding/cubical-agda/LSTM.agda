{-# OPTIONS --cubical #-}

module LSTM where

postulate
  V H : Set

record LSTMState : Set where
  field
    hidden : H
    cell : H

open LSTMState public

record Gates : Set where
  field
    forget : H
    input : H
    output : H
    candidate : H

open Gates public

step : Gates -> LSTMState -> LSTMState
step gates current = record { hidden = output gates ; cell = candidate gates }
