#!/bin/bash
# One task of the B200 campaign on the GPU given by the lane (CUDA_VISIBLE_DEVICES already set). A task is  <exp>:<model>:<arm-or-rep>:
#   f3:af3:1          F3 engine bf16, repetition 1
#   f4:af3:engine     F4 27 targets, arm engine | torch
#   f9:af3:engine     F9 limit search, arm engine | torch
#   f6:af3:miniworld  F6 training, arm miniworld | pytorch | pytorchC | cueqC (3 repeats, model protenix1 | af3)
# Exit code 0 ONLY when the output of the task really exists and is valid (the experiment scripts check it); the lane writes the .done marker from that.
set -uo pipefail
IFS=: read -r EXP MODEL ARG <<<"$1"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$1" in f3:*|f4:*:engine|f9:*:engine|f6:*:miniworld) export B200_ENGINE_FIX=${B200_ENGINE_FIX:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/diag-nan} ;; esac
source "$HERE/common/cards/b200.env"
case $EXP in
  f3) MODEL=$MODEL ARM=engine REP=$ARG exec bash "$HERE/F3_5i28_times/b200.sh" ;;
  f4) MODEL=$MODEL ARM=$ARG exec bash "$HERE/F4_foldbench27_times/b200.sh" ;;
  f9) MODEL=$MODEL ARM=$ARG STEPS="${STEPS:-8 16 20 24 32 40 48 56 64 80 96 128}" exec bash "$HERE/F9_inference_limit/b200.sh" ;;
  f6)
    export MODELS=$MODEL ARMS=$ARG
    bash "$HERE/F6_training_speed/b200.sh"; rc=$?
    OUT=$PAPER_OUT/F6_training_speed/$CARD
    # valid = every repeat has a .done (trained) or a log ending in an OOM line of the last ladder value; anything else is an error
    for r in 1 2 3; do
      ok=0; for f in "$OUT"/${MODEL}_${ARG}_a*_r$r.done; do [ -e "$f" ] && ok=1; done
      if [ $ok = 0 ]; then
        last=$(ls -t "$OUT"/${MODEL}_${ARG}_a*_r$r.log 2>/dev/null | head -1)
        [ -n "$last" ] && grep -aq "^OOM" "$last" && [ "$ARG" != miniworld ] && ok=1
        [ "$ARG" = miniworld ] && [ -n "$last" ] && grep -aq "^OOM" "$last" && ok=1
      fi
      [ $ok = 1 ] || { echo "F6 CHECK repeat $r has neither .done nor OOM"; rc=4; }
    done
    exit $rc ;;
  *) echo "unknown task $1"; exit 2 ;;
esac
