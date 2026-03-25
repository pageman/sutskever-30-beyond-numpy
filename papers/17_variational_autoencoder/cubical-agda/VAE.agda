{-# OPTIONS --cubical #-}

module VAE where

open import Agda.Builtin.Equality

postulate
  Observation Latent Reconstruction : Set

record Encoder : Set1 where
  field
    encode : Observation -> Latent

open Encoder public

record Decoder : Set1 where
  field
    decode : Latent -> Reconstruction

open Decoder public

forward : Encoder -> Decoder -> Observation -> Reconstruction
forward enc dec x = decode dec (encode enc x)

forward-unfold : (enc : Encoder) (dec : Decoder) (x : Observation) -> forward enc dec x ≡ decode dec (encode enc x)
forward-unfold enc dec x = refl
