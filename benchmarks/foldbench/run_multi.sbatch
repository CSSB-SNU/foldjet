#!/bin/bash
# The whole FoldBench (1,522 targets) through FoldJet af3 on N GPUs in ONE job (fb_multi.py: one parent scheduler, workers per GPU):
# bf16, compile + CUDA graphs (scope trunk), 5 trunk_seeds x 5 samples, 200 steps, --recycles 11, trunk_seeds grouped up to 512 AF3 tokens,
# small-target concurrency 3 under CUDA MPS. A walltime measurement is one continuous job: never split it into chunks.
#   sbatch --export=ALL,SPECS=<spec dir>,OUT=<run dir>[,FFENV=<pixi env>][,TARGETS=<list>][,EXTRA="..."] benchmarks/foldbench/run_multi.sbatch
# Resumable: finished targets (<OUT>/<t>/done or failed/marker) are skipped on a resubmit.
# ---- edit these lines for your cluster (partition / account / qos / node, GPUs; -c and --mem scale with the GPU count) ----
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=normal_h100
#SBATCH --gres=gpu:h100:2
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --hint=nomultithread
#SBATCH -J fb-multi
#SBATCH -o slurm-%x-%j.log
# ---- end of the cluster lines ----
set -uo pipefail
: "${SPECS:?SPECS=<spec dir from build_specs.py>}" "${OUT:?OUT=<run dir>}"
REPO=${FOLDJET_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
B=$REPO/benchmarks/foldbench
TARGETS=${TARGETS:-$B/foldbench_targets.txt}
mkdir -p "$OUT"
{ echo "job=${SLURM_JOB_ID:-} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
  git -C "$REPO" log --oneline -1
  echo "start $(date +%s.%N)"; } >> "$OUT/job_info.txt"
# shellcheck disable=SC2086  # EXTRA is a list of flags
bash "$B/run_py.sh" "$B/fb_multi.py" af3 "$TARGETS" "$SPECS" "$OUT" \
  --recycles 11 --group-max-tokens 512 --small-concurrency 3 ${EXTRA:-}
RC=$?
echo "end $(date +%s.%N) rc=$RC" >> "$OUT/job_info.txt"
exit "$RC"
