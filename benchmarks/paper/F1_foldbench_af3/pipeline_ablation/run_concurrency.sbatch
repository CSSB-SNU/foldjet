#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 -w node01 --gres=gpu:h100:1 -c 16 --hint=nomultithread --mem=100G --time=00:10:00 -J f1-conc
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F1/h100/concurrency/logs/%x_%j.log
# F1 concurrency experiment: one size bin (BIN) x one configuration (CFG) on ONE H100, the real FoldBench runner (fb_multi.py, ablation code).
W=/home/psk6950/foldforge-scratch/wt-ablate; O=/home/psk6950/foldforge-scratch/paper/F1/h100/concurrency
. $W/benchmarks/paper/common/env.sh
export FFENV=/home/psk6950/foldjet/.pixi/envs/default FF_ALLOW_SIZE_MISMATCH=1
case "$CFG" in
  k1)      A="--small-concurrency 1 --group 1" ;;
  k1g5)    A="--small-concurrency 1 --group 5" ;;
  k2)      A="--small-concurrency 2 --group 1" ;;
  k3)      A="--small-concurrency 3 --group 1" ;;
  k4)      A="--small-concurrency 4 --group 1" ;;
  k3g5)    A="--small-concurrency 3 --group 5" ;;
  k3nomps) A="--small-concurrency 3 --group 1 --mps off" ;;
  *) echo "CFG?"; exit 2 ;;
esac
D=$O/$BIN/${CFG}_r${REP:-1}; mkdir -p $D
echo "job=$SLURM_JOB_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES bin=$BIN cfg=$CFG rep=${REP:-1} args=$A $(date -Is)" > $D/job_info.txt
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader >> $D/job_info.txt
S=$(date +%s.%N)
bash $W/benchmarks/foldbench/run_py.sh $W/benchmarks/foldbench/fb_multi.py af3 $O/lists/$BIN.txt "$FB_SPECS" $D/pred \
  --recycles 11 --group-max-tokens 768 --small-max-tokens 768 --small-min-left-s 0 $A > $D/run.log 2>&1
RC=$?; E=$(date +%s.%N)
echo "end rc=$RC wall_s=$(awk "BEGIN{printf \"%.1f\", $E - $S}") done=$(ls $D/pred/*/done 2>/dev/null | wc -l) failed=$(ls -d $D/pred/*/failed 2>/dev/null | wc -l)" >> $D/job_info.txt
