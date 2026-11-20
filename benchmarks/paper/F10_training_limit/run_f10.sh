#!/bin/bash
# F10 on ONE card (h100.sbatch / a100.sbatch / rtx5090.sh call this).
#   ARMS="miniworld pytorchC cueqC"   PARTS="a b384 b768"   CHUNKS="0,24,12,6,3,1"   OUT=$PAPER_OUT/F10_training_limit/$CARD
#   SMOKE=1: ONE trial (arm miniworld, 384 tokens / 4,096 atoms, 48 blocks, chunk 1, with checkpointing) = the plumbing check
#   LENGTH_LADDER / BLOCK_LADDER: comma lists replacing the default ladders of bisect.py
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../train_common/env.sh"
ARMS=${ARMS:-"miniworld pytorchC cueqC"}; PARTS=${PARTS:-"a b384 b768"}; CHUNKS=${CHUNKS:-"0,24,12,6,3,1"}
OUT=${OUT:-$PAPER_OUT/F10_training_limit/$CARD}; mkdir -p "$OUT"
export OUT
JOB=${SLURM_JOB_ID:-local$$}
{ echo "card=$CARD host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} date=$(date -Is)"; echo "foldforge_commit=$(git -C "$FF_REPO" rev-parse HEAD) dirty=$(git -C "$FF_REPO" status --porcelain | wc -l)";
  echo "engine_src=${ENGINE_SRC:-installed}"; nvidia-smi -L; } | tee -a "$OUT/meta_$JOB.txt"
if [ "${SMOKE:-0}" = 1 ]; then
  python "$HERE/bisect.py" --part a --arm miniworld --out "$OUT/smoke" --chunks 1 --length-ladder 384 --trial-timeout 560
  exit 0
fi
EXTRA=(); [ -n "${LENGTH_LADDER:-}" ] && EXTRA+=(--length-ladder "$LENGTH_LADDER"); [ -n "${BLOCK_LADDER:-}" ] && EXTRA+=(--block-ladder "$BLOCK_LADDER")
for part in $PARTS; do
  for arm in $ARMS; do
    python "$HERE/bisect.py" --part "$part" --arm "$arm" --out "$OUT" --chunks "$CHUNKS" "${EXTRA[@]}"
  done
done
echo "F10 finished $(date -Is)"
