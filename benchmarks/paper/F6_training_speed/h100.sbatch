#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:1 -c 16 --mem=120G --hint=nomultithread --time=04:00:00 -J f6-h100
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F6_training_speed/h100/slurm_%x_%j.log
# F6 on one H100. Full campaign: sbatch h100.sbatch (3 repeats, 8 runs x 2 models). Smoke: SMOKE=1 MODELS=protenix1 ARMS="miniworld" sbatch --time=00:10:00 h100.sbatch
# ENGINE_SRC=<engine source tree> selects a release candidate in front of the installed engine.
export CARD=h100
exec "$(dirname "$(readlink -f "$0")")/run_f6.sh"
