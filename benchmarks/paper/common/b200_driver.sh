#!/bin/bash
# Start one lane per GPU given (default 0 1 2 3; never 4-7).  nohup bash common/b200_driver.sh <queue> <state dir> 0 1 2 3 &
Q=$1; D=$2; shift 2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$D"
for g in "${@:-0 1 2 3}"; do nohup bash "$HERE/b200_lane.sh" "$g" "$Q" "$D" > "$D/lane$g.log" 2>&1 & echo "lane $g pid $!" | tee -a "$D/pids.txt"; done
wait
