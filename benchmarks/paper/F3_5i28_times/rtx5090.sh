#!/bin/bash
# F3 FoldForge side on the RTX 5090 (cssb-rtx): ONE (model, arm=engine, repetition) per call, under the box's GPU lock.
#   MODEL=af3 ARM=engine REP=1 [SIZES="1 2"] [REPEATS=3] benchmarks/paper/F3_5i28_times/rtx5090.sh [SECONDS]
# SECONDS = the timeout of the run (default 7200). Waits for the GPU lock; RAM of everything together stays under 16 GB (the slice caps it).
# Never build and time at once: the engine's sm120 extensions must already be in the pooled JIT cache (the first call of a new engine builds them: do that
# first with SIZES=1, outside the campaign).
set -uo pipefail
: "${MODEL:?MODEL}"; ARM=${ARM:-engine}
[ "$ARM" = engine_fp8 ] && { echo "FP8 is H100 only"; exit 2; }
SECS=${1:-7200}
HERE=${FF_REPO:-$HOME/ff/FoldForge}/benchmarks/paper
source "$HERE/common/cards/rtx5090.env"
OUT=$PAPER_OUT/F3/$CARD/${ARM}_rep${REP:-1}/$MODEL
mkdir -p "$OUT" "$PAPER_OUT/logs"
cd "$FF_REPO"
exec "$HOME/mwe/gpu.sh" "$SECS" bash -c '
  echo "JOB local host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
  python -u '"$HERE"'/F3_5i28_times/run_ff.py --model '"$MODEL"' --arm '"$ARM"' --out '"$OUT"' --rep '"${REP:-1}"' --repeats '"${REPEATS:-6}"' --sizes '"${SIZES:-1 2 4 8 12 16}"' 2>&1 | grep --line-buffered -v UserWarning
  rc=${PIPESTATUS[0]}; echo "END rc=$rc $(date -Is)"; exit $rc' 2>&1 | tee -a "$PAPER_OUT/logs/f3_${CARD}_${ARM}_rep${REP:-1}_${MODEL}.log"
