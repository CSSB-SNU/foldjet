#!/bin/bash
# the 5I28 kit benchmark, controlled setup: resumable N-GPU work queue over QFILE lines "name|command", one GPU per item.
#   PIN=<k>   every item is pinned (taskset) to its OWN k logical CPUs of the job's allowed set (slot s: CPUs [s*k, (s+1)*k)) and OMP/MKL/OPENBLAS/NUMEXPR threads = k
#   (unset: unpinned, library defaults: the pre-2026-11-11 way)
# Around every item, in the same allocation / cgroup / pinning as the kit: probe.py before and after (python loop, 4096^2 f32 matmul, 2 GiB copy, CPU MHz, load),
# `scontrol show node`, a background nvidia-smi logger every 5 s (SM clock, max SM clock, power, temperature, utilization) and a thread sampler every 60 s
# (nproc, taskset -pc and the thread count of every process of the job: ls /proc/<pid>/task | wc -l).
# usage: sbatch -J NAME -w <node> -c <cpus> [--hint=nomultithread] --gres=gpu:h100:$NG --export=ALL,NG=$NG,QFILE=<file>[,PIN=<k>,DONE=<dir>] queue3.sh   (always --mem)
#SBATCH -p h100 -A cssb --qos=normal_h100 --mem=400G --time=08:00:00 -o %x.%j.out
B=${BENCH_DIR:?set BENCH_DIR}; NG=${NG:-2}; QFILE=${QFILE:?}; DONE=${DONE:-$B/done}; PIN=${PIN:-}
PY=${PY:?set PY to a python with numpy (host probe)}
mkdir -p /tmp/q5i28-$SLURM_JOB_ID $DONE $B/probe
ALLOWED=$($PY -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))")
echo "QUEUE START job=$SLURM_JOB_ID host=$(hostname) NG=$NG PIN=${PIN:-none} qfile=$QFILE done=$DONE $(date -Is)"
echo "ALLOWED_CPUS $(echo $ALLOWED | wc -w): $ALLOWED"; scontrol show job $SLURM_JOB_ID | grep -oE "NumCPUs=[0-9]+|MinCPUsNode=[0-9]+|ThreadSpec=[^ ]*|CPUs/Task=[0-9]+" | tr '\n' ' '; echo
scontrol show node $(hostname) | grep -oE "ThreadsPerCore=[0-9]+|CoresPerSocket=[0-9]+|Sockets=[0-9]+" | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nodeline() { scontrol show node $(hostname) | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+|CPUTot=[0-9]+" | tr '\n' ' '; }
run() {
  IFS='|' read -r name cmd <<< "$1"
  if [ -s $DONE/$name ]; then echo "$(date +%T) SKIP $name (done): $(cat $DONE/$name)"; return; fi
  while :; do for s in $(seq 0 $((NG-1))); do exec 9>/tmp/q5i28-$SLURM_JOB_ID/$s; if flock -n 9; then
    P=$B/probe/$name.$(hostname).$SLURM_JOB_ID
    if [ -n "$PIN" ]; then SET=$(echo $ALLOWED | tr ' ' '\n' | sed -n "$((s*PIN+1)),$((s*PIN+PIN))p" | paste -sd, -); PINCMD="taskset -c $SET"; THR="OMP_NUM_THREADS=$PIN MKL_NUM_THREADS=$PIN OPENBLAS_NUM_THREADS=$PIN NUMEXPR_NUM_THREADS=$PIN"
    else SET=all; PINCMD=""; THR="PIN_NONE=1"; fi
    t0=$(date +%s)
    out=$( (export CUDA_VISIBLE_DEVICES=$s; export $THR
       echo "SLOT $s cpus=$SET pin=${PIN:-none} threads_env=$THR" > $P.txt
       echo "NODE before $(nodeline)" >> $P.txt
       $PINCMD $PY $B/probe.py before >> $P.txt 2>&1
       nvidia-smi -i $s --query-gpu=timestamp,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader -l 5 > $P.gpu.csv 2>&1 & GP=$!
       ( sleep 100; while :; do echo "--- threads $(date +%T)" >> $P.threads.txt
           for p in $(ls /proc | grep -E '^[0-9]+$'); do grep -q "job_${SLURM_JOB_ID}/" /proc/$p/cgroup 2>/dev/null || continue
             n=$(ls /proc/$p/task 2>/dev/null | wc -l); [ "$n" -gt 8 ] && echo "pid=$p threads=$n $(taskset -pc $p 2>/dev/null | cut -d: -f2) $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-90)" >> $P.threads.txt
           done; sleep 60; done ) & TS=$!
       $PINCMD bash -c "echo NPROC \$(nproc) AFFINITY \$(taskset -pc \$\$ | cut -d: -f2) >> $P.txt; $cmd" 2>&1 | tail -1
       kill $TS $GP 2>/dev/null; wait $TS $GP 2>/dev/null
       $PINCMD $PY $B/probe.py after >> $P.txt 2>&1
       echo "NODE after $(nodeline)" >> $P.txt ) 2>&1 | tail -1)
    t1=$(date +%s)
    echo "$(date +%T) gpu$s job=$SLURM_JOB_ID $(hostname) pin=${PIN:-none} $name wall=$((t1-t0))s :: $out"
    case "$out" in KIT*) echo "job=$SLURM_JOB_ID host=$(hostname) gpu=$s pin=${PIN:-none} $out" > $DONE/$name;; esac
    k=${name#kit_}; k=${k%_x*}; n=${name##*_x}; cp $B/$k/logs/x$n.log $P.kitlog 2>/dev/null
    flock -u 9; return; fi; done; sleep 2; done
}
export -f run nodeline; export NG B DONE PY SLURM_JOB_ID PIN ALLOWED
tr "\n" "\0" < $QFILE | xargs -0 -P $NG -I{} bash -c 'run "$@"' _ {}
echo "QUEUE DONE $(date -Is)"
