#!/bin/bash
# F4 on the B200 (host b200, GPUs 0-3, no Slurm): ONE (predictor, arm) per call on the lane's GPU. run_foldbench27.py exits non-zero unless all 27 targets
# have a valid final row (ok / out-of-memory / recorded timeout).
#   MODEL=af3 ARM=engine|torch [ONLY=t1,t2] benchmarks/paper/F4_foldbench27_times/b200.sh
set -uo pipefail
: "${MODEL:?MODEL}" "${ARM:?ARM}"
[ "$ARM" = engine_fp8 ] && { echo "FP8 arm: not on the B200 in this campaign"; exit 2; }
HERE=${FF_REPO:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/ff/foldjet}/benchmarks/paper
source "$HERE/common/cards/b200.env"
cd "$FF_REPO"
echo "JOB local host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)"
python -u "$HERE/F4_foldbench27_times/run_foldbench27.py" "$MODEL" "$ARM" --out "$PAPER_OUT/F4/$CARD" 2>&1 | grep --line-buffered -av "Warning\|warn("
rc=${PIPESTATUS[0]}; echo "END rc=$rc $(date -Is)"; exit $rc
