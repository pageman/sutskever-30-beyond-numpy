{-# OPTIONS --cubical #-}

module VAE where

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
