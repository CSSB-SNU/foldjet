#!/bin/bash
# F1 / F8: the whole FoldBench (1,522 targets) through a FoldJet model on the H100s of ONE job: 5 trunk_seeds x 5 samples, 200 steps, 11 trunk
# passes (AF3), bf16, compile + CUDA graphs (scope trunk), MSA up to 16,384 rows, templates as FoldBench gives them.
# A walltime run is ONE continuous job (never split into chunks); resumable on a resubmit (finished targets are skipped).
#
# LAUNCH=shard (default, like the Anthropic kit): one fb_multi.py process per shard, one shard per GPU (CUDA_VISIBLE_DEVICES=k), the kit's shard split
#                 (shards/s0.txt, s1.txt), no MPS, no small-target concurrency. GPU-h = sum of the per-shard wall times.
# LAUNCH=multi:   one fb_multi.py over all GPUs of the job (shared scheduler, small targets 3 at a time under CUDA MPS). GPU-h = GPUs x wall.
# The run records which one it used (<run>/launch.txt, stamp.json, README of the experiment).
#
#   sbatch -J paper-F1-ff-bf16-shard -o $PAPER_OUT/F1/h100/logs/%x_%j.log --export=ALL,NAME=af3_bf16_shard,PRECISION=bf16,LAUNCH=shard \
#       benchmarks/paper/F1_foldbench_af3/foldjet/run_ff.sbatch
#
# Variables: NAME (required)  EXP (F1; F8 for the fine-tune)  MODEL (af3)  PRECISION (bf16|fp8)  LAUNCH (shard|multi)  SHARDS ("0 1")  TARGETS (a list;
#   default all 1,522; in shard mode intersected with each shard)  CKPT (checkpoint instead of the released weights)  RECYCLES (11; empty = model default)
#   EXTRA (more fb_multi / worker flags, e.g. "--seeds 42" for a smoke)  FFENV  ENGINE_OVERLAY  FOLDFORGE_REPO  FB_SPECS  PAPER_OUT
# GPUs: --gres=gpu:h100:2 below = one per shard / two for multi. Smoke runs override on the command line: --gres=gpu:h100:1 --time=00:10:00 -c 16 --mem=64G SHARDS=0.
#SBATCH -p h100
#SBATCH -A cssb
#SBATCH --qos=cssb_h100
#SBATCH --gres=gpu:h100:2
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH --hint=nomultithread
#SBATCH --time=7-00:00:00
#SBATCH -J paper-ff
#SBATCH -o slurm-%x-%j.log
set -uo pipefail
: "${NAME:?NAME=<run name>}"
REPO=${FOLDFORGE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
P=$REPO/benchmarks/paper
B=$REPO/benchmarks/foldbench
. "$P/common/env.sh"
EXP=${EXP:-F1}; MODEL=${MODEL:-af3}; PRECISION=${PRECISION:-bf16}; LAUNCH=${LAUNCH:-shard}; SHARDS=${SHARDS:-0 1}
RECYCLES=${RECYCLES-11}
TARGETS=${TARGETS:-$B/foldbench_targets.txt}
OUT=${OUT:-$PAPER_OUT/$EXP/h100/$NAME}
FFENV=${FFENV:-$REPO/.pixi/envs/default}; export FFENV
mkdir -p "$OUT"

# --- FP8 (engine 3.x opt-in, process-wide): MINIWORLD_FP8=safe + the model named for TriAttn; lazy guards = no host sync under CUDA-graph capture.
# The worker calls fp8.check() after every trunk_seed group and redoes a saturated group (<= 3 times); the counts are in each run.json ("fp8").
unset MINIWORLD_FP8 MINIWORLD_FP8_MODEL MINIWORLD_TRIMUL_FP8_GUARD MINIWORLD_OPM_FP8_GUARD MINIWORLD_PWA_FP8_GUARD MINIWORLD_TRANSITION_FP8_GUARD
case "$PRECISION" in
  bf16) ;;
  fp8) export MINIWORLD_FP8=safe MINIWORLD_FP8_MODEL=$MODEL MINIWORLD_TRIMUL_FP8_GUARD=lazy MINIWORLD_OPM_FP8_GUARD=lazy MINIWORLD_PWA_FP8_GUARD=lazy MINIWORLD_TRANSITION_FP8_GUARD=lazy ;;
  *) echo "PRECISION must be bf16 or fp8" >&2; exit 2 ;;
esac

ARGS=(--group-max-tokens 512)
[ -n "$RECYCLES" ] && ARGS+=(--recycles "$RECYCLES")
[ -n "${CKPT:-}" ] && ARGS+=(--checkpoint "$CKPT")
# shellcheck disable=SC2206  # EXTRA is a list of flags
ARGS+=(${EXTRA:-})

PY=$FFENV/bin/python
echo "launch=$LAUNCH precision=$PRECISION model=$MODEL name=$NAME exp=$EXP targets_file=$TARGETS shards=($SHARDS) ckpt=${CKPT:-released}" > "$OUT/launch.txt"
{ echo "job=${SLURM_JOB_ID:-} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-}"; git -C "$REPO" log --oneline -1; } >> "$OUT/job_info.txt"
PATH=$FFENV/bin:$PATH LD_LIBRARY_PATH=$FFENV/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=${ENGINE_OVERLAY:+$ENGINE_OVERLAY:}$REPO/src \
  "$PY" "$P/common/stamp.py" "$OUT/stamp.json" "launch=$LAUNCH" "precision=$PRECISION" "model=$MODEL" "name=$NAME" "experiment=$EXP" "checkpoint=${CKPT:-released}" || true

done_count() { ls "$1"/*/done 2>/dev/null | wc -l; }
failed_count() { ls -d "$1"/*/failed 2>/dev/null | wc -l; }

if [ "$LAUNCH" = multi ]; then
  S=$(date +%s.%N)
  echo "start $S $(date -Is)" >> "$OUT/job_info.txt"
  bash "$B/run_py.sh" "$B/fb_multi.py" "$MODEL" "$TARGETS" "$FB_SPECS" "$OUT/pred" "${ARGS[@]}" --small-concurrency 3
  RC=$?
  E=$(date +%s.%N)
  echo "end $E $(date -Is) rc=$RC wall_s=$(awk "BEGIN{printf \"%.1f\", $E - $S}") done=$(done_count "$OUT/pred") failed=$(failed_count "$OUT/pred")" >> "$OUT/job_info.txt"
  exit "$RC"
fi

[ "$LAUNCH" = shard ] || { echo "LAUNCH must be shard or multi" >&2; exit 2; }
# shard mode: shard k -> GPU index = position in SHARDS (CUDA_VISIBLE_DEVICES of the allocation), its own run dir <run>/pred_s<k>
i=0; PIDS=()
for k in $SHARDS; do
  L=$OUT/targets_s$k.txt
  if [ "$TARGETS" != "$B/foldbench_targets.txt" ]; then grep -Fxf "$P/F1_foldbench_af3/shards/s$k.txt" "$TARGETS" > "$L"; else cp "$P/F1_foldbench_af3/shards/s$k.txt" "$L"; fi
  D=$OUT/pred_s$k; mkdir -p "$D"
  (
    GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | cut -d, -f$((i + 1)))
    export CUDA_VISIBLE_DEVICES=$GPU
    S=$(date +%s.%N)
    echo "shard=$k gpu=$GPU host=$(hostname) job=${SLURM_JOB_ID:-} targets=$(wc -l < "$L")" > "$D/job_info.txt"
    echo "start $S $(date -Is)" >> "$D/job_info.txt"
    bash "$B/run_py.sh" "$B/fb_multi.py" "$MODEL" "$L" "$FB_SPECS" "$D" "${ARGS[@]}" --small-concurrency 1
    RC=$?
    E=$(date +%s.%N)
    echo "end $E $(date -Is) rc=$RC wall_s=$(awk "BEGIN{printf \"%.1f\", $E - $S}") done=$(done_count "$D") failed=$(failed_count "$D")" >> "$D/job_info.txt"
  ) > "$OUT/shard_$k.log" 2>&1 &
  PIDS+=($!); i=$((i + 1))
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
exit "$RC"
