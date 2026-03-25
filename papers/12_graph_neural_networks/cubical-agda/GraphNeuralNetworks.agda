{-# OPTIONS --cubical #-}

module GraphNeuralNetworks where

open import Agda.Builtin.Equality

postulate
  Node Message GraphLabel : Set

record MessagePassing : Set1 where
  field
    send : Node -> Message
    aggregate : Message -> Message -> Message

open MessagePassing public

record Readout : Set1 where
  field
    classify : Message -> GraphLabel

open Readout public

forward : MessagePassing -> Readout -> Node -> Node -> GraphLabel
forward mp ro a b = classify ro (aggregate mp (send mp a) (send mp b))

forward-unfold : (mp : MessagePassing) (ro : Readout) (a b : Node) -> forward mp ro a b ≡ classify ro (aggregate mp (send mp a) (send mp b))
forward-unfold mp ro a b = refl
