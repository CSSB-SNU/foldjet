#!/bin/bash
# F8: submit the two whole-FoldBench runs (released Protenix v1, fine-tune epoch 10) and, behind each, its full 25-sample evaluation. Prints the sbatch commands
# it runs. One continuous GPU job per arm (never chunked); the evaluation is a CPU job (--gres=none) that starts when the run succeeded.
#   FOLDFORGE_REPO=<FoldForge checkout at the experiment commit> FFENV=<env> [ENGINE_OVERLAY=<dir>] bash benchmarks/paper/F8_protenix_ft_foldbench/run_f8.sh
# LAUNCH=multi (default here: accuracy only, the wall time of F8 is not a result); FT_CKPT overrides the fine-tune checkpoint. Variables of the two sbatch scripts apply.
set -euo pipefail
REPO=${FOLDFORGE_REPO:?FOLDFORGE_REPO}; P=$REPO/benchmarks/paper
. "$P/common/env.sh"
FT_CKPT=${FT_CKPT:-$FF_SCRATCH/ft_train/runs/ft_engine_fp32d/epoch0010.miniworld.pt}
[ -f "$FT_CKPT" ] || { echo "missing checkpoint $FT_CKPT" >&2; exit 1; }
mkdir -p "$PAPER_OUT/F8/h100/logs"
for arm in released ft10; do
  CK=""; [ "$arm" = ft10 ] && CK=$FT_CKPT
  J=$(sbatch --parsable -J paper-F8-$arm -o "$PAPER_OUT/F8/h100/logs/%x_%j.log" \
      --export=ALL,EXP=F8,NAME=protenix1_$arm,MODEL=protenix1,PRECISION=bf16,LAUNCH=${LAUNCH:-multi},RECYCLES=,CKPT=$CK,FOLDFORGE_REPO=$REPO \
      "$P/F1_foldbench_af3/foldforge/run_ff.sbatch")
  echo "run $arm: job $J"
  E=$(sbatch --parsable --dependency=afterok:$J -J paper-F8-eval-$arm -o "$PAPER_OUT/F8/h100/logs/%x_%j.log" \
      --export=ALL,NAME=protenix1_$arm,PRED_DIRS="$PAPER_OUT/F8/h100/protenix1_$arm/pred",EVAL_OUT="$PAPER_OUT/F8/h100/eval/protenix1_$arm",FOLDFORGE_REPO=$REPO \
      "$P/F2_foldbench_accuracy/eval.sbatch")
  echo "eval $arm: job $E (after $J)"
done
