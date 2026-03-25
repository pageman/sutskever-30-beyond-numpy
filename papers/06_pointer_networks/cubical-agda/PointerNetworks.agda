{-# OPTIONS --cubical #-}

module PointerNetworks where

open import Agda.Builtin.Equality

postulate
  Item Query Position : Set

record Encoder : Set1 where
  field
    encode : Item -> Query

open Encoder public

record Pointer : Set1 where
  field
    point : Query -> Query -> Position

open Pointer public

forward : Encoder -> Pointer -> Item -> Item -> Position
forward e p a b = point p (encode e a) (encode e b)

forward-unfold : (e : Encoder) (p : Pointer) (a b : Item) -> forward e p a b ≡ point p (encode e a) (encode e b)
forward-unfold e p a b = refl
