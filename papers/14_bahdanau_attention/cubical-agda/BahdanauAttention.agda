{-# OPTIONS --cubical #-}

module BahdanauAttention where

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
