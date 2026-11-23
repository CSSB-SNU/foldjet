#!/bin/bash
# F9 on the RTX 5090 (cssb-rtx): ONE (predictor, arm) per call = the whole limit search, under the box's GPU lock (RAM of everything < 16 GB; never build and time at
# once). Every probe is its own process (run_size.py) started INSIDE the lock-held call.
#   MODEL=af3 ARM=engine|torch [STEPS="8 16 24"] benchmarks/paper/F9_inference_limit/rtx5090.sh [SECONDS]      (FP8 is H100 only; no kit install on this box)
set -uo pipefail
: "${ARM:?ARM engine|torch}" "${MODEL:?MODEL}"
[ "$ARM" = engine_fp8 ] && { echo "FP8 is H100 only"; exit 2; }
SECS=${1:-28800}
HERE=${FF_REPO:-$HOME/ff/FoldForge}/benchmarks/paper
source "$HERE/common/cards/rtx5090.env"
export OMP_NUM_THREADS=8
mkdir -p "$PAPER_OUT/logs"; cd "$FF_REPO"
OUT=$PAPER_OUT/F9/$CARD/$ARM/$MODEL
exec "$HOME/mwe/gpu.sh" "$SECS" bash -c '
  echo "JOB local host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
  python -u '"$HERE"'/F9_inference_limit/limit_search.py --model '"$MODEL"' --arm '"$ARM"' --out '"$OUT"' '"${STEPS:+--steps $STEPS}"'
  rc=$?; echo "END rc=$rc $(date -Is)"; exit $rc' 2>&1 | tee -a "$PAPER_OUT/logs/f9_${CARD}_${ARM}_${MODEL}.log"
