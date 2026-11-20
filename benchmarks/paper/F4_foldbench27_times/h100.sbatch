#!/bin/bash
# F4 (and the runs F5 uses) on the H100 cluster: ONE (predictor, arm) per job, the 27 targets ascending in one process, resumable.
#   sbatch -J f4-af3-engine --export=ALL,MODEL=af3,ARM=engine benchmarks/paper/F4_foldbench27_times/h100.sbatch
# ARM: torch | engine | engine_fp8 | anthropic.  Smoke: `--time=00:10:00` and ONLY=7sxp-assembly1,7fxw-assembly1.  Full campaign: submit_all.sh.
# Optional: NODE (pin the node; the arms of one predictor should share it), ENGINE_SRC, PAPER_OUT.
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --hint=nomultithread
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%j.log
set -uo pipefail
: "${MODEL:?MODEL}" "${ARM:?ARM}"
HERE=${FF_REPO:-$HOME/foldforge-scratch/paper-infer}/benchmarks/paper
source "$HERE/common/cards/h100.env"
export OMP_NUM_THREADS=8
echo "JOB $SLURM_JOB_ID host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
cd "$FF_REPO"
python -u "$HERE/F4_foldbench27_times/run_foldbench27.py" "$MODEL" "$ARM" --out "$PAPER_OUT/F4/$CARD" 2>&1 | grep --line-buffered -av "Warning\|warn("
echo "END rc=${PIPESTATUS[0]} $(date -Is)"
