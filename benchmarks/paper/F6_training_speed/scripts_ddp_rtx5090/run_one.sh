#!/bin/bash
# usage: [CFG=...] [PIN=1] [STEPS=30] [VALIDATE=0] run_one.sh TAG GPU_LIST [hydra overrides...]
#   e.g. run_one.sh n4_ga1 0,1,2,3 train.grad_accum_steps=1
# Holds the ~/mwe/gpu<i>.lock of every GPU it uses; per-rank NUMA pinning via rank_exec.sh; results in results/TAG/.
set -u
D=/home/psk6950/mwe/agents/mw_ddp; . $D/env.sh
TAG=$1; GPUS=$2; shift 2
N=$(echo "$GPUS" | tr , '\n' | wc -l)
CFG=${CFG:-$MW_ROOT/configs/miniworld/phase1a_distogram_medium_v240_nr_nt_pf16.yaml}
OUT=$D/results/$TAG; mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPUS PIN=${PIN:-1}
VAL=$([ "${VALIDATE:-0}" = 1 ] && echo "" || echo "--no-validate")
locks=(); for g in ${GPUS//,/ }; do locks+=(flock /home/psk6950/mwe/gpu$g.lock); done
( while sleep 2; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader; done ) > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
echo "=== $TAG gpus=$GPUS n=$N pin=$PIN cfg=$CFG overrides=$* $(date -Is)"
cd /tmp
"${locks[@]}" timeout ${TIMEOUT:-3000} torchrun --standalone --nproc_per_node=$N --no-python $D/rank_exec.sh \
  $D/synth_graph_train.py --config "$CFG" --steps ${STEPS:-30} $VAL --out "$OUT/summary.json" -- \
  train.run_dir=$OUT/run train.use_wandb=false train.train_item=${TRAIN_ITEM:-61440} "$@"
rc=$?
kill $SMI 2>/dev/null
cp $OUT/run/*/*/timing-rank*.json "$OUT/" 2>/dev/null
echo "=== $TAG rc=$rc $(date -Is)"
exit $rc
