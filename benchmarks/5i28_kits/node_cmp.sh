#!/bin/bash
#SBATCH -p h100 -A cssb --qos=normal_h100 -w node02 --gres=gpu:h100:1 -c 120 --mem=300G --time=01:30:00 -J b5i28-nodecmp -o %x.%j.out
# host-contention discriminator: the same n=1 kit runs on the lighter node (node02), CPULoad / clocks recorded around each run. KITS="protenix_v1 af3_torch"
B=${BENCH_DIR:?set BENCH_DIR}
echo "NODECMP START job=$SLURM_JOB_ID host=$(hostname) $(date -Is)"; scontrol show node $(hostname) | grep -oE "CPUAlloc=[0-9]+|CPULoad=[0-9.]+|CPUTot=[0-9]+" | tr '\n' ' '; echo
nvidia-smi -q -d CLOCK -i 0 | grep -E "Graphics|SM |Memory|Product" | head -8
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader
for k in ${KITS:-protenix_v1}; do
  echo "CPULOAD before $k: $(scontrol show node $(hostname) | grep -oE 'CPULoad=[0-9.]+|CPUAlloc=[0-9]+' | tr '\n' ' ')"
  case $k in protenix_v1|protenix_v2|opendde) CMD="bash $B/protenix_family_run.sh $k 1";; *) CMD="bash $B/$k/run.sh 1";; esac
  ( export CUDA_VISIBLE_DEVICES=0; eval "$CMD" 2>&1 | tail -1 ) > $B/logs/nodecmp.$SLURM_JOB_ID.$k.line &
  P=$!; sleep 240; echo "CPULOAD during $k (+240s): $(scontrol show node $(hostname) | grep -oE 'CPULoad=[0-9.]+' | tr '\n' ' ')"; wait $P
  echo "RESULT $(hostname) $k: $(cat $B/logs/nodecmp.$SLURM_JOB_ID.$k.line)"
  echo "CPULOAD after $k: $(scontrol show node $(hostname) | grep -oE 'CPULoad=[0-9.]+' | tr '\n' ' ')"
  cp $B/$k/logs/x1.log $B/$k/logs/x1.node02.log 2>/dev/null
done
echo "NODECMP DONE $(date -Is)"
