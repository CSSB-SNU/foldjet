#!/bin/bash
# F6: whole-model training speed on ONE card (called by h100.sbatch / a100.sbatch / rtx5090.sh; source env first through train_common/env.sh).
#   MODELS="protenix1 af3"   ARMS="miniworld pytorch pytorchC cueqC"   (cueq = cuEquivariance without compile, optional)
#   REPEATS=3  STEPS=6  ACCUM=8
#   AUG_LADDER_ENGINE="0"          diffusion augmentations per chunk of the engine arm (0 = all 48 in one pass)
#   AUG_LADDER_OTHER="0 24"        the other arms: first value that does not run out of memory is the one used (24 = 24 x 2); an
#                                  OOM at a value is a recorded result (the log stays), the next value is a new process
#   OUT=$PAPER_OUT/F6_training_speed/$CARD   POOL_DIR=$OUT/pool (stand-in pool caches, pool_<model>.pt, identical for every arm)
#   SMOKE=1  -> STEPS=2 ACCUM=2 REPEATS=1 (plumbing check only)
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../train_common/env.sh"
MODELS=${MODELS:-"protenix1 af3"}; ARMS=${ARMS:-"miniworld pytorch pytorchC cueqC"}
REPEATS=${REPEATS:-3}; STEPS=${STEPS:-6}; ACCUM=${ACCUM:-8}
if [ "${SMOKE:-0}" = 1 ]; then STEPS=2; ACCUM=2; REPEATS=1; fi
AUG_LADDER_ENGINE=${AUG_LADDER_ENGINE:-"0"}; AUG_LADDER_OTHER=${AUG_LADDER_OTHER:-"0 24"}
OUT=${OUT:-$PAPER_OUT/F6_training_speed/$CARD}; POOL_DIR=${POOL_DIR:-$OUT/pool}
mkdir -p "$OUT" "$POOL_DIR"
POOL=$(cat "$HERE/../train_common/pool_targets.txt")
cd "$FF_REPO" || exit 1
JOB=${SLURM_JOB_ID:-local$$}
{ echo "card=$CARD host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} date=$(date -Is)"; echo "foldforge_commit=$(git -C "$FF_REPO" rev-parse HEAD) dirty=$(git -C "$FF_REPO" status --porcelain | wc -l)";
  echo "engine_src=${ENGINE_SRC:-installed}"; nvidia-smi -L; } | tee -a "$OUT/meta_$JOB.txt"
for rep in $(seq 1 "$REPEATS"); do
 for model in $MODELS; do
  for arm in $ARMS; do
    case $arm in
      miniworld) backend=miniworld; extra=(); ladder=$AUG_LADDER_ENGINE ;;
      pytorch)   backend=pytorch;   extra=(); ladder=$AUG_LADDER_OTHER ;;
      pytorchC)  backend=pytorch;   extra=(--compile all); ladder=$AUG_LADDER_OTHER ;;
      cueq)      backend=cueq;      extra=(); ladder=$AUG_LADDER_OTHER ;;
      cueqC)     backend=cueq;      extra=(--compile all); ladder=$AUG_LADDER_OTHER ;;
      *) echo "unknown arm $arm"; continue ;;
    esac
    for chunk in $ladder; do
      tag=${model}_${arm}_a${chunk}_r${rep}; log=$OUT/$tag.log
      [ -e "$OUT/$tag.done" ] && { echo "skip $tag (done)"; break; }
      echo "=== $tag $(date -Is)"
      { echo "RUNMETA card=$CARD host=$(hostname) slurm_job=$JOB model=$model arm=$arm backend=$backend compile=${extra[*]:-none} aug_chunk=$chunk repeat=$rep steps=$STEPS accum=$ACCUM"
        python scripts/train_protenix1.py --bench --model "$model" --backend "$backend" --recipe protenix --steps "$STEPS" --grad-accum "$ACCUM" \
          --warmup-steps 1 --augment-chunk "$chunk" --bench-bucket --bench-pool "$POOL" --bench-pool-cache "$POOL_DIR/pool_$model.pt" \
          --run-dir "$OUT/runs/$tag" --step-graph "${extra[@]}" 2>&1 | grep -a --line-buffered -v "Warning\|warn"
        echo "rc=${PIPESTATUS[0]} $(date -Is)"; } > "$log"
      # keep the lines the table needs on the console
      grep -a "^BENCHSTEP\|^OOM\|^rc=" "$log" | cut -c1-200
      if grep -aq "^BENCH " "$log" && ! grep -aq "^OOM" "$log"; then touch "$OUT/$tag.done"; break; fi
    done
  done
 done
done
echo "F6 finished $(date -Is)"
