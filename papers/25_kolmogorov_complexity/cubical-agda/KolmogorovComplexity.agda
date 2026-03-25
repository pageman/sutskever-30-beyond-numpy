{-# OPTIONS --cubical #-}

module KolmogorovComplexity where

open import Agda.Builtin.Equality

postulate
  Sequence Description Complexity : Set

record Descriptor : Set1 where
  field
    describe : Sequence -> Description
    measure  : Description -> Complexity

open Descriptor public

measure-self : (d : Descriptor) (s : Sequence) -> measure d (describe d s) ≡ measure d (describe d s)
measure-self d s = refl

cong : {A B : Set} {x y : A} -> (f : A -> B) -> x ≡ y -> f x ≡ f y
cong f refl = refl

measure-cong-description
  : (d : Descriptor)
  -> {s₁ s₂ : Sequence}
  -> describe d s₁ ≡ describe d s₂
  -> measure d (describe d s₁) ≡ measure d (describe d s₂)
measure-cong-description d desc≡ = cong (measure d) desc≡
