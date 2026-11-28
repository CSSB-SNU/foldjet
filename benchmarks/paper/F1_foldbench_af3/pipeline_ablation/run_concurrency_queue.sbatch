#!/bin/bash
# F1 pipeline addendum (2026-11-27): the concurrency x seed-grouping runs as ONE job over NG GPUs of one node — a work queue, one run per GPU at
# a time, each run pinned to its own CPUS_PER_RUN cores. Items = BINS x CFGS x REPS; finished items (job_info.txt "end rc=0") are skipped.
#   NG=4 CPUS_PER_RUN=12 REPS="2 3" sbatch -w node01 --gres=gpu:h100:4 -c 48 run_concurrency_queue.sbatch
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --hint=nomultithread
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH -J f1-conc-q
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F1/h100/concurrency/logs/%x_%j.log
set -uo pipefail
W=/home/psk6950/foldforge-scratch/wt-ablate; O=/home/psk6950/foldforge-scratch/paper/F1/h100/concurrency
. $W/benchmarks/paper/common/env.sh
export FFENV=/home/psk6950/foldjet/.pixi/envs/default FF_ALLOW_SIZE_MISMATCH=1 W O FB_SPECS
NG=${NG:-4}; CPR=${CPUS_PER_RUN:-12}; REPS=${REPS:-2 3}
BINS=${BINS:-a_le256 b_257_384 c_385_512 d_513_768}; CFGS=${CFGS:-k1 k1g5 k2 k3 k4 k3g5 k3nomps}
Q=$O/queue_$SLURM_JOB_ID; mkdir -p $Q
ALLOWED=$(taskset -pc $$ | cut -d: -f2 | tr ',' '\n' | awk -F- '{if (NF==2) for(i=$1;i<=$2;i++) print i; else print $1}' | tr -d ' ' | paste -sd' ')
echo "JOB $SLURM_JOB_ID host=$(hostname) NG=$NG cpus/run=$CPR allowed=[$ALLOWED] reps=[$REPS] $(date -Is)"
for r in $REPS; do for c in $CFGS; do for b in $BINS; do echo "$b $c $r"; done; done; done > $Q/items.txt
export Q NG CPR ALLOWED
run() {
  read -r BIN CFG REP <<< "$1"
  D=$O/$BIN/${CFG}_r$REP
  if grep -q "end rc=0" $D/job_info.txt 2>/dev/null; then echo "SKIP $BIN $CFG r$REP"; return; fi
  case "$CFG" in
    k1) A="--small-concurrency 1 --group 1" ;; k1g5) A="--small-concurrency 1 --group 5" ;;
    k2) A="--small-concurrency 2 --group 1" ;; k3) A="--small-concurrency 3 --group 1" ;; k4) A="--small-concurrency 4 --group 1" ;;
    k3g5) A="--small-concurrency 3 --group 5" ;; k3nomps) A="--small-concurrency 3 --group 1 --mps off" ;;
  esac
  # one Slurm step per run: its own GPU (seen as device 0, as in a 1-GPU job: the MPS daemon and its clients must agree on the
  # device index; CUDA_VISIBLE_DEVICES=2 for both made every MPS client fail "CUDA is unavailable" in job 67773) and CPR cores
    rm -rf $D; mkdir -p $D
    echo "job=$SLURM_JOB_ID host=$(hostname) step=srun cpus=$CPR bin=$BIN cfg=$CFG rep=$REP args=$A $(date -Is)" > $D/job_info.txt
    S=$(date +%s.%N)
    srun --exact -n1 -c "$CPR" --gres=gpu:h100:1 --mem=${MEM_PER_RUN:-40G} bash $W/benchmarks/foldbench/run_py.sh $W/benchmarks/foldbench/fb_multi.py af3 $O/lists/$BIN.txt "$FB_SPECS" $D/pred \
      --recycles 11 --group-max-tokens 768 --small-max-tokens 768 --small-min-left-s 0 $A > $D/run.log 2>&1
    RC=$?; E=$(date +%s.%N)
    echo "end rc=$RC wall_s=$(awk "BEGIN{printf \"%.1f\", $E - $S}") done=$(ls $D/pred/*/done 2>/dev/null | wc -l) failed=$(ls -d $D/pred/*/failed 2>/dev/null | wc -l)" >> $D/job_info.txt
    echo "$(date +%T) $BIN $CFG r$REP rc=$RC $(tail -1 $D/job_info.txt)"
}
export -f run
tr '\n' '\0' < $Q/items.txt | xargs -0 -P "$NG" -I{} bash -c 'run "$@"' _ {}
echo "END $(date -Is)"
