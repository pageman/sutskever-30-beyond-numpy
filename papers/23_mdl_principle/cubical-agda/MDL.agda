{-# OPTIONS --cubical #-}

module MDL where

postulate
  Model Data Cost : Set

record CodeLength : Set1 where
  field
    dataCost : Data -> Cost
    modelCost : Model -> Cost

open CodeLength public
