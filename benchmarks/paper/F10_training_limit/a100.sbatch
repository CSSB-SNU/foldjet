#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 -c 16 --mem=120G --time=12:00:00 -J f10-a100
#SBATCH -o /home/psk6950/paper_out/F10_a100_%x_%j.log
# F10 on one A100 (cssb3; no account / qos). Resumable (results.jsonl).
export CARD=a100
unset SLURM_ACCOUNT
exec "${FF_REPO:-$HOME/FoldJet}/benchmarks/paper/F10_training_limit/run_f10.sh"
