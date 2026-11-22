#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=none -c 16 --mem=32G --time=00:30:00 -J F11tr-down
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F11_investigation/trace/logs/%x_%j.log
R=/home/psk6950/foldforge-scratch/paper-306; E=$R/.pixi/envs/default; T=/home/psk6950/foldforge-scratch/paper/F11_investigation/trace
export PATH=$E/bin:$PATH PYTHONPATH=$R/src
python $T/down.py $T/out/down.csv only_transition only_trimul safe only_diffusion no_transition
