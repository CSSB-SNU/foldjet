#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:1 -c 8 --mem=120G --time=02:00:00 -J F11tr
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F11_investigation/trace/logs/%x_%j.log
# F11 trace: bf16 ref + Transition-only FP8 + schedules in one process per target (see trace.py). TARGETS="t1 t2 ...".
R=/home/psk6950/foldforge-scratch/paper-306; B=$R/benchmarks/foldbench; D=/home/psk6950/foldforge-scratch/paper/F11_investigation; T=$D/trace
export FFENV=$R/.pixi/envs/default ENGINE_OVERLAY=$D/overlay FF_ALLOW_SIZE_MISMATCH=1
export MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=af3 MINIWORLD_TRANSITION_FP8_GUARD=lazy F11INV_TRIATTN=none MINIWORLD_TRIMUL_FP8= MINIWORLD_OPM_FP8= MINIWORLD_PWA_FP8= MINIWORLD_TOKEN_DIT_FP8= MINIWORLD_ATOMDIT_FP8=
nvidia-smi --query-gpu=name --format=csv,noheader
for t in $TARGETS; do
  bash $B/run_py.sh $T/trace.py --target $t --out ${OUT:-$T/out} ${SCHED:+--sched $SCHED} ${REC32:+--rec32} 2>&1 | grep -vE "Warning|warn\(" 
done
echo END
