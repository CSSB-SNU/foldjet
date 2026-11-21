#!/bin/bash
# F5 (b): score one (predictor, arm) of the F4 runs against the experimental structures with the FoldBench evaluator (OpenStructure through its singularity
# image + DockQ v2), CPU only, on the H100 cluster (it has the FoldBench upstream tree and the ost image). Predictions made on the A100 / RTX 5090 are copied
# here first (README). One array task per line of eval_items.txt (predictor arm); smoke: a single task with ONLY targets (see README).
#   RUNS=$PAPER_OUT/F4/h100 CARD=h100 sbatch --array=0-32 benchmarks/paper/F5_sample_comparison/eval.sbatch
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=none
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -J f5-eval
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%A_%a.log
set -uo pipefail
: "${RUNS:?RUNS = the F4 output dir of the card}" "${CARD:?CARD of the predictions}"
HERE=${FF_REPO:-$HOME/foldforge-scratch/paper-306}/benchmarks/paper
source "$HERE/common/cards/h100.env"          # the CPU job runs on the H100 cluster whatever card made the predictions
D=$HERE/F5_sample_comparison
OUT=$PAPER_OUT/F5/$CARD
read -r M A < <(sed -n "$((SLURM_ARRAY_TASK_ID + ${TASK_OFFSET:-0} + 1))p" "${ITEMS:-$D/eval_items.txt}")
[ -f "$RUNS/rows/${A}__${M}.jsonl" ] || { echo "no rows for $A $M in $RUNS"; exit 0; }
E=$OUT/eval/${A}__${M}
FBU=${FOLDBENCH_ROOT:-$FB_ROOT}
PYLIB=$OUT/pylib; mkdir -p "$PYLIB"; [ -e "$PYLIB/parallelbar" ] || ln -s "$(ls -d $FBU/upstream/.venv/lib/python3*/site-packages/parallelbar | head -1)" "$PYLIB/parallelbar"
echo "eval $M $A card=$CARD start=$(date -Is) job=$SLURM_JOB_ID"
python "$D/prepare_eval.py" "$RUNS" "$A" "$M" "$OUT/eval" || exit 1
export PYTHONPATH=$PYLIB:$PYTHONPATH MW_OST_WORKERS=12 MW_DOCKQ_WORKERS=12 FOLDBENCH_ROOT=$FBU
python -u "$D/evaluate_ranked.py" --pred-dir "$E/pred" --eval-dir "$E/eval" --name "${A}__${M}" --python "$(which python)"
echo "end=$(date -Is)"
