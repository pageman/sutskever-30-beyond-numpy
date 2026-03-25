{-# OPTIONS --cubical #-}

module RAG where

open import Agda.Builtin.Equality

postulate
  Query Doc Output : Set

record Retriever : Set1 where
  field
    retrieve : Query -> Doc

record Generator : Set1 where
  field
    generate : Query -> Doc -> Output

open Retriever public
open Generator public

generate-self : (g : Generator) (q : Query) (d : Doc) -> generate g q d ≡ generate g q d
generate-self g q d = refl
