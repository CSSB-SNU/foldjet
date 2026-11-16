#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:2 -c 32 -w node01 --hint=nomultithread --mem=300G --time=1-00:00:00 -J ftengpae
#SBATCH -o %x-%j.log
# Protenix v1 fine-tuning, Protenix recipe (--recipe protenix), PDB only + MiniWorld sampler weights, crop 384, effective batch 256, backend miniworld.
# Checkpoints as MiniWorld: last.miniworld.pt overwritten at the end of every epoch, epoch<E> every 5 epochs. One 24 h segment of a chain; resumes from the newer of last / epoch<E>.
source "${FF_REPO:-$HOME/FoldForge}/scripts/finetune/run/common.sh"
export FT_WORKERS=16 BACKEND=miniworld
nvidia-smi -L
train ${RUN_ROOT:-$HOME/foldforge-scratch/ft_train/runs}/ft_engine_pae ${TOTAL_EPOCHS:-1000}
echo "TRAIN rc=$?"
