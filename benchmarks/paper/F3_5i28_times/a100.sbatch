#!/bin/bash
# F3 FoldJet side on the A100 (cssb3, own Slurm): ONE (model, arm=engine, repetition) per job. FP8 is H100 only.
#   sbatch -J f3-af3 --export=ALL,MODEL=af3,ARM=engine,REP=1 benchmarks/paper/F3_5i28_times/a100.sbatch      (run on cssb3, from the checkout there)
# Smoke: `--time=00:10:00`, SIZES="1 2" REPEATS=3.
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%j.log
set -uo pipefail
: "${MODEL:?MODEL}"; ARM=${ARM:-engine}
[ "$ARM" = engine_fp8 ] && { echo "FP8 is H100 only"; exit 2; }
HERE=${FF_REPO:-$HOME/FoldJet}/benchmarks/paper
source "$HERE/common/cards/a100.env"
OUT=$PAPER_OUT/F3/$CARD/${ARM}_rep${REP:-1}/$MODEL
mkdir -p "$OUT" "$PAPER_OUT/logs"
echo "JOB $SLURM_JOB_ID host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
cd "$FF_REPO"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_clocks.csv" 2>&1 &
GL=$!
python -u "$HERE/F3_5i28_times/run_ff.py" --model "$MODEL" --arm "$ARM" --out "$OUT" --rep "${REP:-1}" --repeats "${REPEATS:-6}" --sizes ${SIZES:-1 2 4 8 12 16} 2>&1 | grep --line-buffered -v UserWarning
RC=${PIPESTATUS[0]}
kill $GL 2>/dev/null
echo "END rc=$RC $(date -Is)"
exit "$RC"
