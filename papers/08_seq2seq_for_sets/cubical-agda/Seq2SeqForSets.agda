{-# OPTIONS --cubical #-}

module Seq2SeqForSets where

open import Agda.Builtin.Equality

postulate
  Item Summary Output : Set

record Pooler : Set1 where
  field
    pool : Item -> Item -> Summary

open Pooler public

record Decoder : Set1 where
  field
    decode : Summary -> Output

open Decoder public

forward : Pooler -> Decoder -> Item -> Item -> Output
forward p d a b = decode d (pool p a b)

forward-unfold : (p : Pooler) (d : Decoder) (a b : Item) -> forward p d a b ≡ decode d (pool p a b)
forward-unfold p d a b = refl
