#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=none -c 4 --mem=16G --time=00:15:00 -J F11tr-an
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F11_investigation/trace/logs/%x_%j.log
E=/home/psk6950/foldforge-scratch/paper-306/.pixi/envs/default; T=/home/psk6950/foldforge-scratch/paper/F11_investigation/trace
export PATH=$E/bin:$PATH
python $T/analyze.py $T/out > $T/out/report_raw.txt 2>&1; echo EXIT $?
