{-# OPTIONS --cubical #-}

module Pruning where

postulate
  Input Hidden Output : Set

record SparseNetwork : Set1 where
  field
    keep  : Input -> Hidden
    pruned : Input -> Hidden
    readout : Hidden -> Output

open SparseNetwork public
