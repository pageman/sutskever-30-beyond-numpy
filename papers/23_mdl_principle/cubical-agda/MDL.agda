{-# OPTIONS --cubical #-}

module MDL where

open import Agda.Builtin.Equality

postulate
  Model Data Cost : Set

record CodeLength : Set1 where
  field
    dataCost : Data -> Cost
    modelCost : Model -> Cost

open CodeLength public

dataCost-self : (c : CodeLength) (d : Data) -> dataCost c d ≡ dataCost c d
dataCost-self c d = refl
