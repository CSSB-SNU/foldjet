#!/bin/bash
# F3 FoldForge side on the B200 (host b200, GPUs 0-3, no Slurm): ONE (model, arm=engine, repetition) per call, on the GPU the lane gave us (CUDA_VISIBLE_DEVICES).
#   MODEL=af3 ARM=engine REP=1 [SIZES="1 2"] [REPEATS=3] benchmarks/paper/F3_5i28_times/b200.sh
# Exit code 0 ONLY when every size has a final cell (ok or oom) in cells.jsonl: run_ff.py itself returns 0 after a `failed` cell.
set -uo pipefail
: "${MODEL:?MODEL}"; ARM=${ARM:-engine}
[ "$ARM" = engine_fp8 ] && { echo "FP8 arm: not on the B200 in this campaign"; exit 2; }
HERE=${FF_REPO:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/ff/foldforge}/benchmarks/paper
source "$HERE/common/cards/b200.env"
OUT=$PAPER_OUT/F3/$CARD/${ARM}_rep${REP:-1}/$MODEL
mkdir -p "$OUT"; cd "$FF_REPO"
echo "JOB local host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date -Is)"
python -u "$HERE/F3_5i28_times/run_ff.py" --model "$MODEL" --arm "$ARM" --out "$OUT" --rep "${REP:-1}" --repeats "${REPEATS:-6}" --sizes ${SIZES:-1 2 4 8 12 16} 2>&1 | grep --line-buffered -av "UserWarning\|warn("
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ]; then
  python - "$OUT/cells.jsonl" "${SIZES:-1 2 4 8 12 16}" <<'PY' || rc=3
import json, sys
need = {int(x) for x in sys.argv[2].split()}
final = {}
for line in open(sys.argv[1]) if __import__("os").path.exists(sys.argv[1]) else []:
    r = json.loads(line)
    if r["status"] in ("ok", "oom"):
        final[r["n"]] = r["status"]
missing = sorted(need - set(final))
print("F3 CHECK final", final, "missing", missing)
sys.exit(1 if missing else 0)
PY
fi
echo "END rc=$rc $(date -Is)"; exit $rc
