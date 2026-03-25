{-# OPTIONS --cubical #-}

module LostInMiddle where

open import Agda.Builtin.Equality

postulate
  Chunk Position Query Output : Set

record Reader : Set1 where
  field
    score : Chunk -> Query -> Position
    read  : Position -> Output

open Reader public

score-self : (r : Reader) (c : Chunk) (q : Query) -> score r c q ≡ score r c q
score-self r c q = refl
