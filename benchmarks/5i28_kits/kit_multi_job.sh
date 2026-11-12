#!/bin/bash
# one kit, ALL n in one process, one single-GPU allocation (-c 16 --hint=nomultithread, pinned to its 16 physical cores). Host probe + GPU clock logger +
# thread sampler around the job (item boundaries are inside the kit process: the logger CSV is time-aligned with the kit log's timestamps).
# usage: sbatch -J NAME --gres=gpu:h100:1 -c 16 --hint=nomultithread --mem=240G --time=03:00:00 --export=ALL,KIT=<kit>[,CMD=<command>] kit_multi_job.sh
#SBATCH -p h100 -A cssb --qos=normal_h100 -o %x.%j.out
B=${BENCH_DIR:?set BENCH_DIR}; KIT=${KIT:?}; PIN=16; PY=${PY:?set PY to a python with numpy (host probe)}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ALLOWED=$($PY -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))")
SET=$(echo $ALLOWED | tr ' ' '\n' | sed -n "1,${PIN}p" | paste -sd, -)
export OMP_NUM_THREADS=$PIN MKL_NUM_THREADS=$PIN OPENBLAS_NUM_THREADS=$PIN NUMEXPR_NUM_THREADS=$PIN
P=$B/probe/multi_$KIT.$(hostname).$SLURM_JOB_ID
nodeline() { scontrol show node $(hostname) | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+|CPUTot=[0-9]+" | tr '\n' ' '; }
echo "JOB START job=$SLURM_JOB_ID host=$(hostname) kit=$KIT cpus=$SET ($(echo $ALLOWED | wc -w) allowed) $(date -Is)"
echo "SLOT cpus=$SET pin=$PIN threads_env=OMP/MKL/OPENBLAS/NUMEXPR=$PIN" > $P.txt; echo "NODE before $(nodeline)" >> $P.txt
taskset -c $SET $PY $B/probe.py before >> $P.txt 2>&1
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -l 5 > $P.gpu.csv 2>&1 & GP=$!
( sleep 120; while :; do echo "--- threads $(date +%T)" >> $P.threads.txt
    for p in $(ls /proc | grep -E '^[0-9]+$'); do grep -q "job_${SLURM_JOB_ID}/" /proc/$p/cgroup 2>/dev/null || continue
      n=$(ls /proc/$p/task 2>/dev/null | wc -l); [ "$n" -gt 8 ] && echo "pid=$p threads=$n $(taskset -pc $p 2>/dev/null | cut -d: -f2) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-90)" >> $P.threads.txt
    done; sleep 120; done ) & TS=$!
t0=$(date +%s)
case $KIT in protenix_v1|protenix_v2|opendde) CMD="bash $B/protenix_family_multi.sh $KIT";; *) CMD=${CMD:?};; esac
taskset -c $SET bash -c "echo NPROC \$(nproc) AFFINITY \$(taskset -pc \$\$ | cut -d: -f2) >> $P.txt; $CMD"
RC=$?
t1=$(date +%s); kill $TS $GP 2>/dev/null; wait $TS $GP 2>/dev/null
taskset -c $SET $PY $B/probe.py after >> $P.txt 2>&1; echo "NODE after $(nodeline)" >> $P.txt
echo "JOB END job=$SLURM_JOB_ID kit=$KIT rc=$RC wall=$((t1-t0))s $(date -Is)"
