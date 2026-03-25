{-# OPTIONS --cubical #-}

module GPipe where

open import Agda.Builtin.Equality

postulate
  Input Hidden Output : Set

record Stage1 : Set1 where
  field
    run1 : Input -> Hidden

open Stage1 public

record Stage2 : Set1 where
  field
    run2 : Hidden -> Output

open Stage2 public

forward : Stage1 -> Stage2 -> Input -> Output
forward s1 s2 x = run2 s2 (run1 s1 x)

forward-unfold : (s1 : Stage1) (s2 : Stage2) (x : Input) -> forward s1 s2 x ≡ run2 s2 (run1 s1 x)
forward-unfold s1 s2 x = refl
