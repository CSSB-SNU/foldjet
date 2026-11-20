#!/bin/bash
# One F10 trial = one process. trial.sh ARM TOKENS ATOMS CHUNK CKPT(1|0) BLOCKS(0=native 48) LOG
# ARM: miniworld | pytorch | pytorchC | cueq | cueqC.  Protenix v1, Protenix recipe, MSA depth 1,024, step graph, 4 recycles (the worst case),
# 3 optimizer steps of 1 micro step, stand-in pool padded to TOKENS / ATOMS (the shape sets the memory). Exit code 0 always; the log is the record.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../train_common/env.sh"
ARM=$1 L=$2 A=$3 CHUNK=$4 CKPT=$5 NB=$6 LOG=$7
case $ARM in
  miniworld) backend=miniworld; extra=() ;;
  pytorch)   backend=pytorch;   extra=() ;;
  pytorchC)  backend=pytorch;   extra=(--compile all) ;;
  cueq)      backend=cueq;      extra=() ;;
  cueqC)     backend=cueq;      extra=(--compile all) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
[ "$CKPT" = 1 ] || extra+=(--no-activation-checkpointing)
[ "$NB" != 0 ] && export PF_BLOCKS=$NB
OUT=${OUT:-$PAPER_OUT/F10_training_limit/$CARD}; POOL_DIR=${POOL_DIR:-$OUT/pool}
mkdir -p "$POOL_DIR" "$(dirname "$LOG")"
cd "$FF_REPO" || exit 1
{ echo "TRIALMETA card=$CARD host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} arm=$ARM tokens=$L atoms=$A aug_chunk=$CHUNK ckpt=$CKPT blocks=$NB $(date -Is)"
  ${TRIAL_WRAPPER:-} python "$HERE/../train_common/trial_main.py" --bench --model protenix1 --backend "$backend" --recipe protenix --steps 3 --grad-accum 1 --warmup-steps 1 \
    --augment-chunk "$CHUNK" --bench-bucket --bench-tokens "$L" --bucket-atoms "$A" --bench-pool "$(cat "$HERE/../train_common/pool_targets.txt")" \
    --bench-pool-cache "$POOL_DIR/pool_protenix1.pt" --msa-rows 1024 --recycles 4 --run-dir "$OUT/runs/trial" --step-graph "${extra[@]}" 2>&1 | grep -a --line-buffered -v "Warning\|warn"
  echo "rc=${PIPESTATUS[0]} $(date -Is)"; } > "$LOG"
exit 0
