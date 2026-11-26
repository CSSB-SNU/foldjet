#!/bin/bash
# F9 on the H100 cluster: ONE (predictor, arm) per job = the whole limit search (steps 16 20 24 32 ... then bisection) in ONE continuous job; resumable (probes
# already in sizes.jsonl are not repeated). Arms: engine | engine_fp8 | torch; kit: ARM=kit KIT=<kit> (the Anthropic kit through the 5I28 kit harness, KIT_BENCH).
#   sbatch -J f9-af3-engine --export=ALL,MODEL=af3,ARM=engine benchmarks/paper/F9_inference_limit/h100.sbatch
# Smoke: `--time=00:10:00` and STEPS="1 2" (a size list; the search logic itself is tested with a fake probe: see README).  Full campaign: submit_all.sh.
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%j.log
set -uo pipefail
: "${ARM:?ARM engine|engine_fp8|torch|kit}"
HERE=${FF_REPO:-$HOME/FoldJet}/benchmarks/paper
source "$HERE/common/cards/h100.env"
export OMP_NUM_THREADS=8
echo "JOB $SLURM_JOB_ID host=$(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader
cd "$FF_REPO"
if [ "$ARM" = kit ]; then : "${KIT:?KIT}"; WHO=$KIT; ARGS=(--kit "$KIT"); else : "${MODEL:?MODEL}"; WHO=$MODEL; ARGS=(--model "$MODEL" --arm "$ARM"); fi
OUT=$PAPER_OUT/F9/$CARD/$ARM/$WHO
python -u "$HERE/F9_inference_limit/limit_search.py" "${ARGS[@]}" --out "$OUT" ${STEPS:+--steps $STEPS} ${PROBE_AFTER:+--probe-after $PROBE_AFTER}
echo "END rc=$? $(date -Is)"
