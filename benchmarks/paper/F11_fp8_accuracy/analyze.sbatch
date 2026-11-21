#!/bin/bash
# F11 analysis stages (CPU only, --gres=none): STAGE=rmsd | analyze   (README.md)
#   sbatch -J paper-F11-rmsd -o $PAPER_OUT/F11/h100/logs/%x_%j.log --export=ALL,STAGE=rmsd benchmarks/paper/F11_fp8_accuracy/analyze.sbatch
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=none
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -J paper-F11
#SBATCH -o slurm-%x-%j.log
set -uo pipefail
: "${STAGE:?STAGE=rmsd|analyze}"
REPO=${FOLDFORGE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
. "$REPO/benchmarks/paper/common/env.sh"
E=${EVAL_ENV:-$FF_SCRATCH/ff_2512/.pixi/envs/default}
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
python -u "$REPO/benchmarks/paper/F11_fp8_accuracy/analyze.py" "$STAGE" --workers "${SLURM_CPUS_PER_TASK:-8}" ${OUT:+--out "$OUT"} ${EVALS:+--evals "$EVALS"}
