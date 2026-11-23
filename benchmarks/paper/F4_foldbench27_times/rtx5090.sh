#!/bin/bash
# F4 on the RTX 5090 (cssb-rtx): ONE (predictor, arm) per call under the box's GPU lock (RAM of everything together < 16 GB; nvcc <= 4; never build and time at once).
#   MODEL=af3 ARM=engine [ONLY=t1,t2] benchmarks/paper/F4_foldbench27_times/rtx5090.sh [SECONDS]      (ARM: torch | engine | anthropic; FP8 is H100 only)
set -uo pipefail
: "${MODEL:?MODEL}" "${ARM:?ARM}"
[ "$ARM" = engine_fp8 ] && { echo "FP8 is H100 only"; exit 2; }
SECS=${1:-14400}
HERE=${FF_REPO:-$HOME/ff/FoldForge}/benchmarks/paper
source "$HERE/common/cards/rtx5090.env"
mkdir -p "$PAPER_OUT/logs"; cd "$FF_REPO"
export OMP_NUM_THREADS=8
exec "$HOME/mwe/gpu.sh" "$SECS" bash -c '
  echo "JOB local host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
  python -u '"$HERE"'/F4_foldbench27_times/run_foldbench27.py '"$MODEL $ARM"' --out '"$PAPER_OUT/F4/$CARD"' 2>&1 | grep --line-buffered -av "Warning\|warn("
  rc=${PIPESTATUS[0]}; echo "END rc=$rc $(date -Is)"; exit $rc' 2>&1 | tee -a "$PAPER_OUT/logs/f4_${CARD}_${ARM}_${MODEL}.log"
