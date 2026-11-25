#!/bin/bash
#SBATCH -p A100 --gres=gpu:A100:1 -c 16 --mem=120G --time=08:00:00 -J f6-a100
#SBATCH -o /home/psk6950/paper_out/F6_a100_%x_%j.log
# F6 on one A100 (cssb3; no account / qos there: unset SLURM_ACCOUNT). FoldJet copy and engine source on cssb3, see README.
export CARD=a100
unset SLURM_ACCOUNT
exec "${FF_REPO:-$HOME/FoldForge}/benchmarks/paper/F6_training_speed/run_f6.sh"
