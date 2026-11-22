#!/bin/bash
# One lane = one GPU of the B200 (GPUs 0-3 only). Takes tasks from the queue file in order, claims each with an atomic mkdir, skips tasks with a .done marker.
#   b200_lane.sh <gpu 0-3> <queue file> <state dir>
# A done marker is written ONLY when the task exits 0 (its own validity check passed); a failed task gets fail/<name> (stays claimed: no automatic retry).
# A task whose weights are not complete yet (common/b200_ready.sh) is skipped in this pass; up to 40 passes, 10 min apart while something is waiting.
# Never starts on a GPU that has > 2 GiB in use (a foreign process): waits 10 min x 6, then releases the task.
GPU=$1; Q=$2; D=$3
case $GPU in 0|1|2|3) ;; *) echo "GPU $GPU not ours (0-3 only)"; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$D/claim" "$D/done" "$D/fail" "$D/logs"
export CUDA_VISIBLE_DEVICES=$GPU
for pass in $(seq 1 40); do
  waiting=0
  while read -r task; do
    [ -z "$task" ] && continue; case $task in \#*) continue ;; esac
    name=${task//:/_}; model=$(cut -d: -f2 <<<"$task")
    [ -e "$D/done/$name" ] && continue
    [ -e "$D/STOP" ] && { echo "lane $GPU: STOP file"; exit 0; }
    [ -d "$D/claim/$name" ] && continue
    bash "$HERE/common/b200_ready.sh" "$model" || { waiting=1; continue; }
    mkdir "$D/claim/$name" 2>/dev/null || continue
    for i in 1 2 3 4 5 6; do
      used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "${used:-99999}" -lt 2048 ] && break
      echo "lane $GPU: GPU busy ($used MiB), waiting ($i/6) $(date -Is)"; sleep 600
    done
    if [ "${used:-99999}" -ge 2048 ]; then echo "lane $GPU: GPU stayed busy, releasing $task"; rmdir "$D/claim/$name"; continue; fi
    echo "=== lane $GPU START $task $(date -Is)"
    bash "$HERE/common/b200_task.sh" "$task" > "$D/logs/$name.log" 2>&1; rc=$?
    echo "=== lane $GPU END $task rc=$rc $(date -Is)"
    if [ $rc = 0 ]; then date -Is > "$D/done/$name"; else echo "$rc $(date -Is)" > "$D/fail/$name"; fi
  done < "$Q"
  [ $waiting = 0 ] && break
  sleep 600
done
echo "lane $GPU finished $(date -Is)"
