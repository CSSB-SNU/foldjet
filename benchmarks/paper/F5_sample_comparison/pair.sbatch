#!/bin/bash
# F5 (a) + the CSVs: sample k against sample k of the PyTorch arm (and PyTorch against itself as the baseline), then F5_<card>_{pairs_summary,gt,gt_summary}.csv.
# CPU only. Run after eval.sbatch finished for the ground-truth part (EVAL_ROOT defaults to $PAPER_OUT/F5/<card>/eval when it exists).
#   RUNS=$PAPER_OUT/F4/h100 CARD=h100 sbatch benchmarks/paper/F5_sample_comparison/pair.sbatch
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=none
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH -J f5-pair
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%j.log
set -uo pipefail
: "${RUNS:?RUNS = the F4 output dir of the card}" "${CARD:?CARD}"
HERE=${FF_REPO:-$HOME/foldforge-scratch/paper-306}/benchmarks/paper
PRED_CARD=$CARD; source "$HERE/common/cards/h100.env"
CARD=$PRED_CARD   # h100.env sets CARD=h100: keep the card that made the predictions (outputs go to F5/<that card>)
D=$HERE/F5_sample_comparison; OUT=$PAPER_OUT/F5/$CARD; mkdir -p "$OUT"
ENGINE=$(grep -ho '"engine_version_installed": "[^"]*"' "$RUNS"/rows/engine__*.jsonl 2>/dev/null | sort -u | sed 's/.*: "//;s/"//' | paste -sd';')
COMMIT=$(grep -ho '"foldforge_commit": "[^"]*"' "$RUNS"/rows/engine__*.jsonl 2>/dev/null | sort -u | head -1 | sed 's/.*: "//;s/"//' | cut -c1-9)
ARMS=${ARMS:-engine,engine_fp8}
python "$D/pair.py" "$RUNS" "$OUT/pairs.csv" --arms "$ARMS" --workers 12 --models "${MODELS:-af3,boltz2,chai1,esmfold2,intellifold2,opendde,openfold3,openfold3-preview2,protenix1,protenix2,rosettafold3}"
python "$D/collect.py" --card "$CARD" --prefix "$D/data/F5_$CARD" --pairs "$OUT/pairs.csv" ${EVAL_ROOT:+--eval-root "$EVAL_ROOT"} \
  $( [ -z "${EVAL_ROOT:-}" ] && [ -d "$OUT/eval" ] && echo --eval-root "$OUT/eval" ) --engine-version "$ENGINE" --commit "$COMMIT"
