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

cong₂ : {A B C : Set} {x x' : A} {y y' : B} -> (f : A -> B -> C) -> x ≡ x' -> y ≡ y' -> f x y ≡ f x' y'
cong₂ f refl refl = refl

forward-cong
  : (attn : Attention)
  -> (dec : Decoder)
  -> (h₁ h₂ : EncoderState)
  -> (s₁ s₂ : DecoderState)
  -> attend attn h₁ s₁ ≡ attend attn h₂ s₂
  -> s₁ ≡ s₂
  -> forward attn dec h₁ s₁ ≡ forward attn dec h₂ s₂
forward-cong attn dec h₁ h₂ s₁ s₂ ctx≡ state≡ = cong₂ (decode dec) ctx≡ state≡
