#!/bin/bash
# F1, kit arm: the whole FoldBench through the Anthropic af3_torch kit (DeepMind AF3 weights), ONE kit process per GPU over its shard (shards/s0.txt, s1.txt:
# the split used for every earlier kit run), both shards in this one job. Fast mode, big mode only for the targets fast cannot do. One continuous job.
#   sbatch -J paper-F1-kit -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3kit,KIT_ROOT=...,KIT_INPUTS=...,CUDA13TK=... benchmarks/paper/F1_foldbench_af3/kit/run_kit.sbatch
# Variables: NAME  KIT_ROOT (kit installs; see benchmarks/5i28_kits/README.md)  KIT_INPUTS (AF3 JSONs, build_inputs.py)  CUDA13TK  FF_SCRATCH (cu130run.sh)
#   SHARDS ("0 1")  SHARD_LIST_DIR (default shards/)  BIG (1)  PAPER_OUT.  Smoke: --gres=gpu:h100:1 --time=00:10:00 SHARDS=0 SHARD_LIST_S0=<2-target list>.
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:2
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH --hint=nomultithread
#SBATCH --time=7-00:00:00
#SBATCH -J paper-F1-kit
#SBATCH -o slurm-%x-%j.log
set -uo pipefail
: "${NAME:?NAME=<run name>}"
REPO=${FOLDJET_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
P=$REPO/benchmarks/paper
. "$P/common/env.sh"
export KIT_ROOT=${KIT_ROOT:-$FF_SCRATCH/anthropic-kits} KIT_INPUTS=${KIT_INPUTS:-$FF_SCRATCH/foldbench/af3full/inputs_all}
export CUDA13TK=${CUDA13TK:-$FF_SCRATCH/cuda13tk/.pixi/envs/default}
export OUT=${OUT:-$PAPER_OUT/F1/h100/$NAME}
export CU130_EXTRA_LD=$KIT_ROOT/libstdcxx:$CUDA13TK/lib
mkdir -p "$OUT"
echo "JOB ${SLURM_JOB_ID:-} node $(hostname) start $(date +%s.%N)" | tee -a "$OUT/job_info.txt"
git -C "$REPO" log --oneline -1 >> "$OUT/job_info.txt"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader >> "$OUT/job_info.txt"
echo "launch=one kit process per shard, one shard per GPU; mode fast, big for fast failures only; shards=(${SHARDS:-0 1})" > "$OUT/launch.txt"
i=0
for k in ${SHARDS:-0 1}; do
  L=$P/F1_foldbench_af3/shards/s$k.txt; v=SHARD_LIST_S$k; [ -n "${!v:-}" ] && L=${!v}
  GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | cut -d, -f$((i + 1)))
  # cu130run.sh runs the container with --cleanenv (only a whitelist of variables passes): hand ours over through env
  CUDA_VISIBLE_DEVICES=$GPU bash "$KIT_ROOT/cu130run.sh" env KIT_ROOT="$KIT_ROOT" KIT_INPUTS="$KIT_INPUTS" OUT="$OUT" CUDA13TK="$CUDA13TK" SHARD_LIST="$L" BIG="${BIG:-1}" \
    bash "$P/F1_foldbench_af3/kit/kit_shard.sh" "$k" > "$OUT/gpu$k.log" 2>&1 &
  i=$((i + 1))
done
wait
echo "JOB end $(date +%s.%N)" >> "$OUT/job_info.txt"
