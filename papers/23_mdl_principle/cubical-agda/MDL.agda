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

cong : {A B : Set} {x y : A} -> (f : A -> B) -> x ≡ y -> f x ≡ f y
cong f refl = refl

dataCost-cong
  : (c : CodeLength)
  -> {d₁ d₂ : Data}
  -> d₁ ≡ d₂
  -> dataCost c d₁ ≡ dataCost c d₂
dataCost-cong c d≡ = cong (dataCost c) d≡
