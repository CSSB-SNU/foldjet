#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:1 -c 16 --mem=120G --hint=nomultithread --time=12:00:00 -J f10-h100
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F10_training_limit/h100/slurm_%x_%j.log
# F10 on one H100. Resumable: results.jsonl is read back, so a second submission continues where a deadline stopped the first.
# Smoke: SMOKE=1 sbatch --time=00:10:00 --export=ALL h100.sbatch
export CARD=h100
exec "${FF_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}/benchmarks/paper/F10_training_limit/run_f10.sh"
