#!/usr/bin/env sh
set -eu

# On this macOS/Python 3.9 setup, the default tinygrad dependency chain tries to
# build pyobjc Metal bindings that do not match the available SDK headers.
# Installing tinygrad itself without those optional deps is enough for the repo's
# LLVM-backed tinygrad implementations.

python3 -m pip install tinygrad==0.9.2 --no-deps

