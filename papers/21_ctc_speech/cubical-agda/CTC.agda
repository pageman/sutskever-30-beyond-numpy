{-# OPTIONS --cubical #-}

module CTC where

postulate
  Alignment Symbol Sequence : Set

record Collapse : Set1 where
  field
    collapse : Alignment -> Sequence

open Collapse public

record Decoder : Set1 where
  field
    emit : Sequence -> Symbol

open Decoder public

forward : Collapse -> Decoder -> Alignment -> Symbol
forward ctc dec a = emit dec (collapse ctc a)
