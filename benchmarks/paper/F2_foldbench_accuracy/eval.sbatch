#!/bin/bash
# F2 (also F8, F11): score FoldJet / kit FoldBench predictions with the upstream FoldBench evaluator, CPU only (no GPU: --gres=none).
#   sbatch -J paper-F2-eval-af3-bf16 -o $PAPER_OUT/F2/h100/logs/%x_%j.log \
#     --export=ALL,NAME=af3_bf16_shard,PRED_DIRS="$PAPER_OUT/F1/h100/af3_bf16_shard/pred_s0:$PAPER_OUT/F1/h100/af3_bf16_shard/pred_s1" \
#     benchmarks/paper/F2_foldbench_accuracy/eval.sbatch
# Variables: NAME (required)  PRED_DIRS (prediction dirs separated by ":" (a shard run has pred_s0:pred_s1); required)  EVAL_OUT (default $PAPER_OUT/F2/h100/<NAME>)  TARGETS_FILE (only these assemblies)
#   TOP1_ONLY=1 (score only the top-ranked sample: faster, no oracle)  EVAL_ENV (python env with numpy / pandas / biopython / networkx)  PARALLELBAR (dir with parallelbar)
#   FB_ROOT / FB_UPSTREAM (see evaluate.py)  MW_OST_WORKERS / MW_DOCKQ_WORKERS (default 30).
# A whole FoldBench run (38,050 predictions, 25 per target) took ~4.5 h on 32 CPUs; --top1-only is ~10x faster.
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=none
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH -J paper-eval
#SBATCH -o slurm-%x-%j.log
set -uo pipefail
: "${NAME:?NAME=<run name>}" "${PRED_DIRS:?PRED_DIRS=<dirs>}"
REPO=${FOLDJET_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
P=$REPO/benchmarks/paper
. "$P/common/env.sh"
EVAL_ENV=${EVAL_ENV:-$HOME/foldjet/.pixi/envs/default}
export PATH=$EVAL_ENV/bin:$PATH LD_LIBRARY_PATH=$EVAL_ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PARALLELBAR:-$HOME/foldjet-engine/paper/pylib}
export MW_OST_WORKERS=${MW_OST_WORKERS:-30} MW_DOCKQ_WORKERS=${MW_DOCKQ_WORKERS:-30}
EVAL_OUT=${EVAL_OUT:-$PAPER_OUT/F2/h100/$NAME}
ARGS=(); for d in ${PRED_DIRS//:/ }; do ARGS+=(--pred-dir "$d"); done
[ -n "${TARGETS_FILE:-}" ] && ARGS+=(--targets-file "$TARGETS_FILE")
[ -n "${TOP1_ONLY:-}" ] && ARGS+=(--top1-only)
mkdir -p "$EVAL_OUT"
{ echo "eval job=${SLURM_JOB_ID:-} host=$(hostname) name=$NAME start=$(date -Is)"; echo "foldjet $(git -C "$REPO" rev-parse HEAD)"; echo "pred_dirs=$PRED_DIRS"; echo "FB_ROOT=${FB_ROOT:-default} FB_UPSTREAM=${FB_UPSTREAM:-default}"; } >> "$EVAL_OUT/eval_info.txt"
python -u "$P/F2_foldbench_accuracy/evaluate.py" "${ARGS[@]}" --eval-dir "$EVAL_OUT" --name "$NAME" --python "$(which python)" 2>&1 | grep --line-buffered -v "it/s\]\|s/it\]"
RC=${PIPESTATUS[0]}
echo "end=$(date -Is) rc=$RC" >> "$EVAL_OUT/eval_info.txt"
exit "$RC"
