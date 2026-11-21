#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=none -c 32 --mem=96G --time=03:00:00 -J F11inv-s1
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F11_investigation/logs/%x_%j.log
E=/home/psk6950/foldforge-scratch/paper-306/.pixi/envs/default
export PATH=$E/bin:$PATH OMP_NUM_THREADS=1
S=/home/psk6950/foldforge-scratch/paper/F11_investigation; cd $S/scripts
python -u step1.py ${STAGES:-wholefb} --out $S/out --workers 32
