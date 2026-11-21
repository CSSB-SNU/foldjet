#!/bin/bash
# FALLBACK only (the default is ONE continuous job, run_ft.sbatch): the shape of the earlier chain, 24 h segments that resume from the newer of
# last / epoch<E> (the loss of a segment end is at most the epoch in progress). Usage: submit_chain.sh BACKEND SEGMENTS   (login node: submits only)
B=${1:?miniworld|pytorch}; N=${2:?segments}
HERE=$(cd "$(dirname "$0")" && pwd); prev=""
for i in $(seq 1 "$N"); do
  dep=(); [ -n "$prev" ] && dep=(--dependency=afterany:$prev)
  prev=$(BACKEND=$B sbatch --parsable -J f7-$B --time=1-00:00:00 "${dep[@]}" --export=ALL "$HERE/run_ft.sbatch")
  echo "segment $i job $prev"
done
