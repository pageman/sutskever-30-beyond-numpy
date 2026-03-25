{-# OPTIONS --cubical #-}

module BahdanauAttention where

open import Agda.Builtin.Equality

postulate
  EncoderState DecoderState Context Output : Set

record Attention : Set1 where
  field
    attend : EncoderState -> DecoderState -> Context

open Attention public

record Decoder : Set1 where
  field
    decode : Context -> DecoderState -> Output

open Decoder public

forward : Attention -> Decoder -> EncoderState -> DecoderState -> Output
forward attn dec h s = decode dec (attend attn h s) s

forward-unfold : (attn : Attention) (dec : Decoder) (h : EncoderState) (s : DecoderState) -> forward attn dec h s ≡ decode dec (attend attn h s) s
forward-unfold attn dec h s = refl
