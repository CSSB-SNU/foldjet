#!/bin/bash
# F3 FoldForge side on the H100 cluster: ONE (model, arm, repetition) per job, all sizes in one process. Three repetitions = three different jobs.
#   sbatch -J f3-af3-engine --export=ALL,MODEL=af3,ARM=engine,REP=1 benchmarks/paper/F3_5i28_times/h100.sbatch
# Smoke: add `--time=00:10:00` and SIZES="1 2" REPEATS=3.   Full campaign: submit_all.sh.
# Optional: NODE (pin the node: kits and FoldForge of one comparison run on the same node), ENGINE_SRC (release-candidate source), PAPER_OUT.
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 16
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%j.log
set -uo pipefail
: "${MODEL:?MODEL}" "${ARM:?ARM engine|engine_fp8}"
HERE=${FF_REPO:-$HOME/foldforge-scratch/paper-306}/benchmarks/paper
source "$HERE/common/cards/h100.env"
OUT=$PAPER_OUT/F3/$CARD/${ARM}_rep${REP:-1}/$MODEL
mkdir -p "$OUT"
echo "JOB $SLURM_JOB_ID host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
cd "$FF_REPO"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -l 5 > "$OUT/gpu_clocks.csv" 2>&1 &
GL=$!
python -u "$HERE/F3_5i28_times/run_ff.py" --model "$MODEL" --arm "$ARM" --out "$OUT" --rep "${REP:-1}" --repeats "${REPEATS:-6}" --sizes ${SIZES:-1 2 4 8 12 16} 2>&1 | grep --line-buffered -v UserWarning
RC=${PIPESTATUS[0]}
kill $GL 2>/dev/null
echo "END rc=$RC $(date -Is)"
exit "$RC"
