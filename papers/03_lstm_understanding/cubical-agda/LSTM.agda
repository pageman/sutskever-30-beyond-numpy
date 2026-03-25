{-# OPTIONS --cubical #-}

module LSTM where

open import Agda.Builtin.Equality

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

step-hidden : (gates : Gates) (current : LSTMState) -> hidden (step gates current) ≡ output gates
step-hidden gates current = refl
