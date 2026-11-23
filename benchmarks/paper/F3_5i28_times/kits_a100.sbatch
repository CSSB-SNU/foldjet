#!/bin/bash
# F3 kit side on the A100 (cssb3): ONE (kit, repetition) per job; the sizes run one after the other in this one job (queue3.sh with one GPU: host probe,
# GPU clock log and thread census around every item). Three repetitions = three different jobs (done markers per repetition).
#   sbatch -J f3k-af3_torch --export=ALL,KIT=af3_torch,REP=1 benchmarks/paper/F3_5i28_times/kits_a100.sbatch
# Smoke: `--time=00:10:00` and SIZES="1" (kit n=1 is a few minutes). Needs the kit installs of the harness (KIT_BENCH, see common/cards/a100.env): they are NOT in
# the repository. Kits are never modified; kit-native modes only (fast, big on fast OOM), no other retry.
#SBATCH -p A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -o /home/psk6950/foldforge-scratch/paper/logs/%x_%j.log
set -uo pipefail
: "${KIT:?KIT}"
HERE=${FF_REPO:-$HOME/FoldForge}/benchmarks/paper
source "$HERE/common/cards/a100.env"
R=$PAPER_OUT/F3/$CARD/kits_rep${REP:-1}
export DONE=$R/done_final BENCH_DIR=$KIT_BENCH B=$KIT_BENCH
mkdir -p "$DONE" "$R/queues"
export KIT_BENCH
bash "$HERE/F3_5i28_times/kits/queue_lines.sh" "$KIT" ${SIZES:-1 2 4 8 12 16} > "$R/queues/$KIT.txt"
cat "$R/queues/$KIT.txt"
export QFILE=$R/queues/$KIT.txt PIN=8   # the A100 harness copy of queue3.sh (no host probe: no python on the host side there)
echo "JOB $SLURM_JOB_ID host=$(hostname) kit=$KIT rep=${REP:-1} $(date -Is)"
bash "$KIT_BENCH/queue_a100.sh"
echo "END $(date -Is)"
