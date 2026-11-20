#!/bin/bash
# F4 / F5 full campaign: one job per (predictor, arm). Run from the card's checkout once, after the release is pinned. Arms: torch engine (+ engine_fp8 on the H100).
#   H100: CARD=h100 NODE=node02 benchmarks/paper/F4_foldbench27_times/submit_all.sh      A100 (cssb3): CARD=a100 ...      5090: see README (shell loop under the GPU lock)
# A refused submit stops the script (no other job shape is substituted: ask).  ARMS="torch engine" overrides.
set -euo pipefail
CARD=${CARD:?CARD=h100|a100}
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS="af3 boltz2 chai1 esmfold2 intellifold2 opendde openfold3 openfold3-preview2 protenix1 protenix2 rosettafold3"
ARMS=${ARMS:-"torch engine"}; [ "$CARD" = h100 ] && ARMS=${ARMS_H100:-"torch engine engine_fp8"}
SB=(); [ -n "${NODE:-}" ] && SB+=(-w "$NODE")
for m in $MODELS; do for a in $ARMS; do
  sbatch "${SB[@]}" -J "f4-$a-$m" --export=ALL,MODEL=$m,ARM=$a "$HERE/$CARD.sbatch"
done; done
