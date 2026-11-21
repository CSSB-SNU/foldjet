#!/bin/bash
# F3 full campaign: every (predictor, arm, repetition) as its own job; kits the same. Run from the checkout of the card, ONCE, after the release is pinned.
#   H100:  CARD=h100 NODE=node02 benchmarks/paper/F3_5i28_times/submit_all.sh            (FoldForge bf16 + FP8 + the 12 kits)
#   A100:  CARD=a100 benchmarks/paper/F3_5i28_times/submit_all.sh                          (on cssb3)
#   5090:  CARD=rtx5090 ... see README (a sequential shell loop under the GPU lock; there is no scheduler)
# WHAT=ff|kits|all (default all), REPS="1 2 3". A refused submit stops the script (no other job shape is substituted: ask).
set -euo pipefail
CARD=${CARD:?CARD=h100|a100}; REPS=${REPS:-"1 2 3"}; WHAT=${WHAT:-all}
HERE=$(cd "$(dirname "$0")" && pwd)
MODELS="protenix1 protenix2 opendde esmfold2 esmfold2-fast intellifold2 af3 boltz2 chai1 openfold3 rosettafold3"
KITS="af3_jax af3_torch openfold3 rf3 chai1 esmfold2 esmfold2_fast boltz2 protenix_v1 protenix_v2 opendde intellifold2_notxla"
SB=()
[ -n "${NODE:-}" ] && SB+=(-w "$NODE")
ARMS="engine"; [ "$CARD" = h100 ] && ARMS="engine engine_fp8"
declare -A KT=([af3_jax]=03:00:00 [af3_torch]=03:00:00 [openfold3]=03:00:00 [rf3]=03:00:00 [chai1]=06:00:00 [esmfold2]=03:00:00 [esmfold2_fast]=03:00:00 [boltz2]=03:00:00 [protenix_v1]=04:00:00 [protenix_v2]=04:00:00 [opendde]=05:00:00 [intellifold2_notxla]=04:00:00)
for rep in $REPS; do
  if [ "$WHAT" != kits ]; then for arm in $ARMS; do for m in $MODELS; do
    sbatch "${SB[@]}" -J "f3-$arm-$m-r$rep" --export=ALL,MODEL=$m,ARM=$arm,REP=$rep "$HERE/$CARD.sbatch"
  done; done; fi
  if [ "$WHAT" != ff ]; then for k in $KITS; do
    sbatch "${SB[@]}" -J "f3k-$k-r$rep" --time=${KT[$k]} --export=ALL,KIT=$k,REP=$rep "$HERE/kits_$CARD.sbatch"
  done; fi
done
