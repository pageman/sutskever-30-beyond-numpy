{-# OPTIONS --cubical #-}

module Complexodynamics where

postulate
  State Structure : Set

record Flow : Set1 where
  field
    evolve : State -> Structure

open Flow public
