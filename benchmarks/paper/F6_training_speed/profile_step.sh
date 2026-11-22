#!/bin/bash
# F6 step profile on ONE card: the F6 engine arm (run_f6.sh protocol: --bench pool, step graph, 48 augmentations in one pass) under torch.profiler, ONE warm optimizer step.
#   CARD=h100|b200 (env.sh)   MODELS="protenix1"  (one model)   ARM=miniworld   STEPS=4 ACCUM=8 PROF_STEP=1   EAGER=1 (no --step-graph: per-op attribution)
#   PROBE=1 (default) runs the GEMM-rate probe first.   OUT=<dir>   POOL_DIR=<pool dir with pool_<model>.pt>   ENGINE_SRC / FFENV / FF_REPO as in run_f6.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../train_common/env.sh"
MODEL=${MODEL:-protenix1}; ARM=${ARM:-miniworld}; STEPS=${STEPS:-4}; ACCUM=${ACCUM:-8}; PROF_STEP=${PROF_STEP:-1}
case $ARM in miniworld) backend=miniworld; extra=();; pytorch) backend=pytorch; extra=();; cueqC) backend=cueq; extra=(--compile all);; *) echo "arm $ARM"; exit 2;; esac
OUT=${OUT:-$PAPER_OUT/F6_profile/$CARD}; POOL_DIR=${POOL_DIR:-$PAPER_OUT/F6_training_speed/$CARD/pool}
TAG=${TAG:-_${MODEL}_${ARM}${EAGER:+_eager}}
mkdir -p "$OUT"
POOL=$(cat "$HERE/../train_common/pool_targets.txt")
cd "$FF_REPO" || exit 1
{ echo "card=$CARD host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} date=$(date -Is) gpu=${CUDA_VISIBLE_DEVICES:-all}"; echo "foldforge_commit=$(git -C "$FF_REPO" rev-parse HEAD) dirty=$(git -C "$FF_REPO" status --porcelain | wc -l)"
  echo "engine_src=${ENGINE_SRC:-installed}"; nvidia-smi -L; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; } | tee "$OUT/meta${TAG}.txt"
if [ "${PROBE:-1}" = 1 ]; then python "$HERE/profile_step.py" --probe 2>&1 | grep -a "^PROBE" | tee "$OUT/probe${TAG}.txt"; fi
sg=(--step-graph); [ -n "${EAGER:-}" ] && sg=()
python "$HERE/profile_step.py" --prof-step "$PROF_STEP" --out "$OUT" --tag "$TAG" --keep-trace -- \
  --bench --model "$MODEL" --backend "$backend" --recipe protenix --steps "$STEPS" --grad-accum "$ACCUM" \
  --warmup-steps 1 --augment-chunk 0 --bench-bucket --bench-pool "$POOL" --bench-pool-cache "$POOL_DIR/pool_$MODEL.pt" \
  --run-dir "$OUT/runs/${MODEL}_${ARM}" "${sg[@]}" "${extra[@]}" 2>&1 | grep -a --line-buffered -v "Warning\|warn" | tee "$OUT/log${TAG}.txt" | grep -a "^BENCHSTEP\|^STEP_GRAPH\|^F6PROF\|^===\|^rc=\|^PRECISION\|^ENGINE_AFTER_MICRO1\|Error\|Traceback" | cut -c1-260
echo "profile finished $(date -Is)"
