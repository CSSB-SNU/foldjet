#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:2 -c 32 -w node01 --hint=nomultithread --mem=128G --time=7-00:00:00
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F7_finetune_loss/h100/slurm_%x_%j.log
# F7: Protenix v1 fine-tuning on 2 x H100, ONE continuous job. BACKEND=miniworld (engine) | pytorch (compile + step graph), set on the command line:
#   BACKEND=miniworld TOTAL_EPOCHS=10 sbatch -J f7-engine --export=ALL --time=3-00:00:00 run_ft.sbatch
#   BACKEND=pytorch   TOTAL_EPOCHS=10 sbatch -J f7-pytorch --export=ALL --time=7-00:00:00 run_ft.sbatch
# Both runs use the same recipe (--recipe protenix, PAE on, precision af3_default: bf16 trunk, fp32 diffusion head and fp32 pair-bias logits),
# the same data stream (FT_SEED, per-rank positions), effective batch 256, 1 epoch = 25,600 samples = 100 optimizer steps, 48 augmentations
# in 2 chunks of 24, activation checkpointing. TOTAL_EPOCHS is the SAME for both (the PyTorch run is never stopped early).
# --mem 128G: the Slurm per-job cap (the earlier chain asked for 300G; RSS is logged in the smoke, see README).
# ENGINE_SRC: engine source tree in front of the installed engine (release candidate). RUN_ROOT: where the run directories go.
set -u
BACKEND=${BACKEND:?set BACKEND=miniworld or pytorch}
RUN_ROOT=${RUN_ROOT:-$HOME/foldforge-scratch/paper/F7_finetune_loss/h100/runs}
export FF_REPO=${FF_REPO:-$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)}
# environment of the earlier fine-tune chain (warm caches; pixi env of the 3.0.1 worktree; pyarrow for the MiniWorld loader)
export FFENV=${FFENV:-$HOME/foldforge-scratch/ff_301/.pixi/envs/default} FT_EXTRA_SITE=${FT_EXTRA_SITE:-$HOME/foldforge-scratch/ft_train/extra_site}
export FT_CONF_CACHE=${FT_CONF_CACHE:-$HOME/foldforge-scratch/ft_train/conf_cache}
P=$HOME/foldforge-scratch/foldbench/cache250
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$P/xdg} TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$P/torch_extensions}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$HOME/foldforge-scratch/train-compile-cache/inductor} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$HOME/foldforge-scratch/train-compile-cache/triton}
mkdir -p "$RUN_ROOT"
[ -n "${ENGINE_SRC:-}" ] && export PYTHONPATH=$ENGINE_SRC/src${PYTHONPATH:+:$PYTHONPATH}
source "$FF_REPO/scripts/finetune/run/common.sh"
export FT_WORKERS=${FT_WORKERS:-16}
[ "$BACKEND" = miniworld ] && NAME=ft_engine || NAME=ft_pytorch
{ echo "F7 meta: backend=$BACKEND host=$(hostname) slurm_job=${SLURM_JOB_ID:-none} date=$(date -Is)"
  echo "foldforge_commit=$(git -C "$FF_REPO" rev-parse HEAD) dirty=$(git -C "$FF_REPO" status --porcelain | wc -l) engine_src=${ENGINE_SRC:-installed}"
  echo "effective=$EFFECTIVE epoch_samples=$EPOCH_SAMPLES total_epochs=${TOTAL_EPOCHS:-?} aug_chunk=${AUG_CHUNK:-24} seed=$FT_SEED"
  nvidia-smi -L; } | tee "$RUN_ROOT.meta_${NAME}_${SLURM_JOB_ID:-local}.txt"
export BACKEND
train "$RUN_ROOT/$NAME" "${TOTAL_EPOCHS:?set TOTAL_EPOCHS (10 = 1,000 steps)}"
echo "TRAIN rc=$?"
