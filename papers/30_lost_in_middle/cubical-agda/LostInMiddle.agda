{-# OPTIONS --cubical #-}

module LostInMiddle where

postulate
  Chunk Position Query Output : Set

record Reader : Set1 where
  field
    score : Chunk -> Query -> Position
    read  : Position -> Output

open Reader public
