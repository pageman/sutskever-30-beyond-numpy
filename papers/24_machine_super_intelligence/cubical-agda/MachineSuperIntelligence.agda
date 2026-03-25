{-# OPTIONS --cubical #-}

module MachineSuperIntelligence where

postulate
  Capability Consequence : Set

record Aggregator : Set1 where
  field
    assess : Capability -> Consequence

open Aggregator public
