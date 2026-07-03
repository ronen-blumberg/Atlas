#!/usr/bin/env bash
# Build Atlas for Linux 64-bit.
# Requires FreeBASIC (fbc) in PATH.  The gcc backend at -O3 is much faster
# than the default backend for the training inner loops.
#   -mt  : multi-threaded runtime (REQUIRED -- training uses threads)
set -e
cd "$(dirname "$0")"
fbc -gen gcc -O 3 -mt -Wc -march=native,-funroll-loops atlas.bas -x atlas
echo "built ./atlas"
echo "  ./atlas train [steps]   train from data/corpus.txt -> model.bin"
echo "  ./atlas chat            chat with the trained model"
