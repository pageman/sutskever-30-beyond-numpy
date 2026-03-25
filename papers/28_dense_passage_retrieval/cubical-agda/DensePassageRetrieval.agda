{-# OPTIONS --cubical #-}

module DensePassageRetrieval where

postulate
  Query Passage Score : Set

record Encoder : Set1 where
  field
    encodeQ : Query -> Score
    encodeP : Passage -> Score

open Encoder public
