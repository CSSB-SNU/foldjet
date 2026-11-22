#!/bin/bash
# F6 step profile of the ENGINE arm on ONE free B200 GPU (host b200, no Slurm). Same environment as the campaign's F6 engine task (common/b200_task.sh):
#   B200_ENGINE_FIX3=$B200_P/diag-nan2 (engine main f06154bc7 tree + its own sm100a JIT cache; CUDA_HOME = nvcc 12.9 so the cache is not judged stale), common/cards/b200.env.
# Usage:  GPU=<0-3> [EAGER=1] [MODEL=protenix1] bash b200_profile.sh      (refuses to start when ANY compute process is on the GPU, and marks the result
#         CONTAMINATED when a foreign process shows up during the run: a contaminated profile is never a result).
set -u
GPU=${GPU:?set GPU=0..3}
case $GPU in 0|1|2|3) ;; *) echo "GPU must be 0-3 (4-7 belong to the colleague)"; exit 2;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
apps() { nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null; }
if [ -n "$(apps)" ]; then echo "GPU $GPU busy: $(apps | tr '\n' ' ') -> not started"; exit 3; fi
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader); echo "GPU $GPU before: $used"
export CUDA_VISIBLE_DEVICES=$GPU CARD=b200 B200_ENGINE_FIX3=${B200_ENGINE_FIX3:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/diag-nan2}
export OUT=${OUT:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/ff/paper/F6_profile/b200}
mkdir -p "$OUT"
( prev=0; while sleep 10; do n=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -vc "^$"); [ "$n" -gt 1 ] && echo "foreign compute process seen $(date -Is): $(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')" >> "$OUT/CONTAMINATED_gpu$GPU"; done ) &
WATCH=$!
bash "$HERE/profile_step.sh"; rc=$?
kill $WATCH 2>/dev/null
[ -e "$OUT/CONTAMINATED_gpu$GPU" ] && echo "CONTAMINATED: a foreign process was on the GPU during the run; discard"
exit $rc
