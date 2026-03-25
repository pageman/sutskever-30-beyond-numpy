{-# OPTIONS --cubical #-}

module RAG where

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
