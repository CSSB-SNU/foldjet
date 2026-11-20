# source this: the Protenix v1 fine-tune (Protenix recipe, PAE on, MiniWorld v2.4.0 data, 2 ranks, effective batch 256).
#   - 1 epoch = 25600 samples = 100 optimizer steps; last.miniworld.pt overwritten at the end of every epoch, epoch<E>.miniworld.pt every 5 epochs
#   - precision af3_default (the protenix recipe default): bf16 trunk on fp32 masters, the diffusion head in fp32
#   - AUG_CHUNK 24: the 48 augmentations in two chunks (fp32 diffusion + compile + step graph at chunk 0 runs out of memory on 80 GB)
#   - BACKEND=pytorch (compile + step graph) or BACKEND=miniworld (engine kernels + step graph; compile measured: no gain)
#   - same data stream for both backends (FT_SEED, per-rank positions)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache} MAX_JOBS=${MAX_JOBS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FT_DATA_CONFIG=${FT_DATA_CONFIG:-$FT_ADAPTER/data_20261014_local_ft.yaml}
export FT_PUMP=${FT_PUMP:-1} FT_WORKERS=${FT_WORKERS:-16} FT_SEED=${FT_SEED:-0} FT_LOG_EVERY=${FT_LOG_EVERY:-25}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CKPT0=${CKPT0:-$HOME/.cache/foldforge/checkpoints/protenix/protenix1.miniworld.pt}
EFFECTIVE=${EFFECTIVE:-256}
EPOCH_SAMPLES=${EPOCH_SAMPLES:-25600}
SPE=$((EPOCH_SAMPLES / EFFECTIVE))

# train RUN_DIR TOTAL_EPOCHS [extra args...]: 2 ranks; resumes from the newer of last.* and epoch*
train() {
  local run=$1 epochs=$2; shift 2
  mkdir -p "$run"
  local best=0 bestfile="" s f e
  if [ -f "$run/last.step" ] && [ -f "$run/last.miniworld.pt" ]; then best=$(cat "$run/last.step"); bestfile=$run/last.miniworld.pt; fi
  for f in "$run"/epoch*.miniworld.pt; do
    [ -e "$f" ] || continue
    e=${f##*epoch}; e=${e%%.*}; s=$((10#$e * SPE))
    if [ "$s" -gt "$best" ]; then best=$s; bestfile=$f; fi
  done
  local resume=()
  [ -n "$bestfile" ] && resume=(--resume "$bestfile")
  local mode=(--backend "${BACKEND:-pytorch}" --step-graph)
  [ "${BACKEND:-pytorch}" = pytorch ] && mode+=(--compile all)
  echo "TRAIN run=$run backend=${BACKEND:-pytorch} epochs=$epochs steps_per_epoch=$SPE effective=$EFFECTIVE start_step=$best resume=${bestfile:-none} $(date -Is)"
  export FT_START_STEP=$best
  cd "$FF_REPO"
  python -u -m torch.distributed.run --standalone --nproc-per-node 2 scripts/train_protenix1.py \
    --recipe protenix --batch-source ft_adapter:batches --model protenix1 --ckpt "$CKPT0" "${mode[@]}" \
    --activation-checkpointing --augment-chunk "${AUG_CHUNK:-24}" \
    --effective-batch "$EFFECTIVE" --epoch-samples "$EPOCH_SAMPLES" --epochs "$epochs" \
    --save-every-epochs 5 --save-last \
    --run-dir "$run" --seed "${FT_SEED}" "${resume[@]}" "$@"
}
