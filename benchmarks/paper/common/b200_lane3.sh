#!/bin/bash
# B200 lane, GPUs 0-7 (0-3 always ours; 4-7 approved by the user 2026-11-22 and shared with a colleague who may start training: yield at once).
#   b200_lane3.sh <gpu 0-7> <queue file> <state dir>        STOPFILE (default $D/STOP_v3) ends the lane after the current task
# - Before every task: any compute process on the GPU (the lane is idle then, so it is foreign) -> GPUs 4-7: STOP this lane and report (marker $D/stopped_gpu<g>);
#   GPUs 0-3: wait 10 min x 6, then release the task.
# - During a task: a watcher samples the GPU's compute processes every 20 s; a pid that is not a descendant of this lane = foreign. A task that saw a foreign
#   process is 'external-contention', never a result: output purged (common/b200_purge.sh), claim removed (requeued), reason in $D/contention/<task>.
#   (Out-of-memory with a foreign process present is therefore not recorded as OOM.)
# - done marker only when the task exits 0 and saw no foreign process; fail marker otherwise (no automatic retry).
GPU=$1; Q=$2; D=$3; STOPFILE=${STOPFILE:-$D/STOP_v3}
case $GPU in 0|1|2|3|4|5|6|7) ;; *) echo "bad GPU $GPU"; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$D/claim" "$D/done" "$D/fail" "$D/logs" "$D/contention"
export CUDA_VISIBLE_DEVICES=$GPU
LANE_PID=$$
apps() { nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '; }
mine() { local p=$1; while [ -n "$p" ] && [ "$p" != 1 ] && [ "$p" != 0 ]; do [ "$p" = "$LANE_PID" ] && return 0; p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null); done; return 1; }
for pass in $(seq 1 40); do
  waiting=0
  while read -r task; do
    [ -z "$task" ] && continue; case $task in \#*) continue ;; esac
    name=${task//:/_}; model=$(cut -d: -f2 <<<"$task")
    [ -e "$D/done/$name" ] && continue
    [ -e "$STOPFILE" ] && { echo "lane $GPU: STOP file"; exit 0; }
    [ -d "$D/claim/$name" ] && continue
    [ -e "$D/fail/$name" ] && continue
    bash "$HERE/common/b200_ready.sh" "$model" || { waiting=1; continue; }
    mkdir "$D/claim/$name" 2>/dev/null || continue
    ok=0
    for i in 1 2 3 4 5 6; do
      if [ -z "$(apps)" ]; then ok=1; break; fi
      if [ "$GPU" -ge 4 ]; then echo "lane $GPU: foreign process on the GPU ($(apps | tr '\n' ' ')): lane stops $(date -Is)"; rmdir "$D/claim/$name"; date -Is > "$D/stopped_gpu$GPU"; exit 0; fi
      echo "lane $GPU: GPU busy, waiting ($i/6) $(date -Is)"; sleep 600
    done
    if [ $ok = 0 ]; then echo "lane $GPU: GPU stayed busy, releasing $task"; rmdir "$D/claim/$name"; continue; fi
    echo "=== lane $GPU START $task $(date -Is)"
    rm -f "$D/logs/$name.foreign"
    ( while sleep 20; do for p in $(apps); do mine "$p" || echo "$p $(date -Is)" >> "$D/logs/$name.foreign"; done; done ) &
    WATCH=$!
    bash "$HERE/common/b200_task.sh" "$task" > "$D/logs/$name.log" 2>&1; rc=$?
    kill $WATCH 2>/dev/null; wait $WATCH 2>/dev/null
    echo "=== lane $GPU END $task rc=$rc $(date -Is)"
    if [ -s "$D/logs/$name.foreign" ]; then
      cp "$D/logs/$name.foreign" "$D/contention/$name"; echo "rc=$rc lane=$GPU" >> "$D/contention/$name"
      bash "$HERE/common/b200_purge.sh" "$task" >> "$D/logs/$name.log" 2>&1
      rmdir "$D/claim/$name" 2>/dev/null; rm -rf "$D/claim/$name"
      echo "lane $GPU: $task external-contention, purged and requeued"
      [ "$GPU" -ge 4 ] && { date -Is > "$D/stopped_gpu$GPU"; echo "lane $GPU: stopping (foreign process)"; exit 0; }
      continue
    fi
    if [ $rc = 0 ]; then date -Is > "$D/done/$name"; else echo "$rc $(date -Is)" > "$D/fail/$name"; fi
  done < "$Q"
  [ $waiting = 0 ] && break
  sleep 600
done
echo "lane $GPU finished $(date -Is)"
