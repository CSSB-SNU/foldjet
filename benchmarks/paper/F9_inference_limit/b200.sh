#!/bin/bash
# F9 on the B200 (host b200, GPUs 0-3, no Slurm): ONE (predictor, arm) = the whole limit search on the lane's GPU; each probe is its own process (run_size.py).
# Exit 0 only when limit.json exists and the search did not stop on a non-memory failure (stop_reason empty).
#   MODEL=af3 ARM=engine|torch [STEPS="8 16 24"] benchmarks/paper/F9_inference_limit/b200.sh
set -uo pipefail
: "${ARM:?ARM engine|torch}" "${MODEL:?MODEL}"
[ "$ARM" = engine_fp8 ] && { echo "FP8 arm: not on the B200 in this campaign"; exit 2; }
HERE=${FF_REPO:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/ff/foldjet}/benchmarks/paper
source "$HERE/common/cards/b200.env"
export OMP_NUM_THREADS=8
cd "$FF_REPO"
OUT=$PAPER_OUT/F9/$CARD/$ARM/$MODEL
echo "JOB local host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)"
python -u "$HERE/F9_inference_limit/limit_search.py" --model "$MODEL" --arm "$ARM" --out "$OUT" ${STEPS:+--steps $STEPS}
rc=$?
if [ "$rc" = 0 ]; then
  python - "$OUT/limit.json" <<'PY' || rc=3
import json, os, sys
if not os.path.exists(sys.argv[1]):
    print("F9 CHECK no limit.json"); sys.exit(1)
s = json.load(open(sys.argv[1]))
print("F9 CHECK", {k: s.get(k) for k in ("largest_ok_contiguous", "first_not_ok", "first_not_ok_status", "stop_reason", "non_monotonic")})
sys.exit(1 if s.get("stop_reason") else 0)
PY
fi
echo "END rc=$rc $(date -Is)"; exit $rc
