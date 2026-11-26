#!/bin/bash
# The 5I28 benchmark, Anthropic openfold3 kit = OpenFold3 0.4.1 + preview-2 weights (~/foldforge-scratch/anthropic-kits/openfold3, read-only), ONE n on the GPU the caller
# gives (CUDA_VISIBLE_DEVICES). Real MSA + 4 distinct real templates (make_inputs.sh), 4 trunk cycles (OF3 num_recycles=3 + 1), 200 steps,
# 5 samples, kit precision of the mode. One process runs COPIES identical queries = COPIES forwards, the first cold.
# usage: run.sh N        env: MODE=fast|exact|off (default fast = the kit mode compared so far), COPIES (default 4 at n=1, else 3)
# Full log: logs/x<N>.log. Stdout: ONE line
#   KIT openfold3 n=<N> exit=<rc> fwd=[<s>,...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
# fwd = the kit's `PHASE item=.. total_s=` (trunk + sampler + confidence, CUDA-synchronised; = its `Model forward time` span, audit
#       docs/records/anthropic_kits_fwd_definition_audit_20261029.md), in item order; warm_median = median of all but the first.
# templates_real = min over items of the kit's `TEMPLATES FEATURISED query=.. real_slots=k/T` (the featurised batch's real slots).
# msa_rows = the MSA depth of the model's feature batch, from the kit's `[of3_graphs] ... msa[1, 1, R, ..]` line (fast/exact; `?` otherwise).
set -uo pipefail
N=${1:?usage: run.sh N}
D=/home/psk6950/miniworld-engine-scratch/bench5i28_1118_steady/openfold3_p2
K=/home/psk6950/foldforge-scratch/anthropic-kits
KD=$K/openfold3
MODE=${MODE:-fast}; SFX=$([ "$MODE" = fast ] || echo _$MODE)
COPIES=${COPIES:-$([ "$N" = 1 ] && echo 4 || echo 3)}
KIT=openfold3_p2; [ "$MODE" = fast ] || KIT=openfold3_p2:$MODE
R=$D/runs/x$N$SFX${RUNTAG:-}; LOG=$D/logs/x$N$SFX${RUNTAG:-}.log   # RUNTAG: concurrent repetitions of one size get their own run dir / log
rm -rf "$R"; mkdir -p "$R/out" "$R/tlogs" "$D/logs" "$D/tmp" /tmp/ffk_of3p2${RUNTAG:-} "$D/cache/openfold_cache"
if false; then :
else "$D/make_inputs.sh" "$N" "$COPIES" "$R/many.json"; fi
cat > "$R/runner.yml" <<EOF
template_preprocessor_settings:
  structure_directory: $D/inputs/template_cifs
  structure_file_format: cif
  fetch_missing_structures: false
  output_directory: $R/template_data
  create_logs: true
  log_directory: $R/tlogs
EOF
T13=/home/psk6950/foldforge-scratch/cuda13tk/.pixi/envs/default   # the kits' toolchain on this cluster (as ../openfold3/run.sh): conda gcc + CUDA 13 compat libcuda in the Ubuntu 22.04 container
export CC=$T13/bin/x86_64-conda-linux-gnu-gcc CXX=$T13/bin/x86_64-conda-linux-gnu-g++ CUDA_HOME=$T13 CU130_EXTRA_LD=$T13/lib \
       TRITON_LIBCUDA_PATH=/home/psk6950/foldforge-scratch/cudacompat13/usr/local/cuda-13.0/compat
export PATH=$KD/venv/bin:$T13/bin:$PATH
export XDG_CACHE_HOME=$D/cache TRITON_CACHE_DIR=$D/cache/triton TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TORCH_EXTENSIONS_DIR=$D/cache/torch_ext \
       MODEL_OPT_JIT_ROOT=$D/cache/jit HF_HOME=$D/cache/hf TORCH_HOME=$D/cache/torch TMPDIR=/tmp/ffk_of3p2${RUNTAG:-} PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENFOLD3_CKPT=$K/weights/openfold3/of3-p2-155k.pt OPENFOLD_CACHE=$D/cache/openfold_cache
unset PYTHONPATH
{
  echo "BENCH 5I28 kit=openfold3(preview2, 0.4.1) mode=$MODE n=$N tokens=$((128 * N)) copies=$COPIES host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "BENCH gpu: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
  echo "BENCH input: $R/many.json (msa $D/inputs/msa/A/colabfold_main.a3m, $(grep -c '^>' "$D/inputs/msa/A/colabfold_main.a3m") a3m rows; templates 5syd_A 6kol_A 5yt7_A 6tfd_A CIF-direct)"
} > "$LOG"
cd "$KD"
timeout --kill-after=60 7200 "$D/cu130kit.sh" bash run.sh pred --config ${KIT_CONFIG:-h100} --mode "$MODE" --query-json "$R/many.json" --output-dir "$R/out" \
  --use-msa-server false --use-templates true --upstream-fix OF3-001 --num-diffusion-samples 5 --runner-yaml "$R/runner.yml" >> "$LOG" 2>&1
RC=$?
# ---- evidence from the kit's own log ----
FWD=$(grep -oE 'PHASE item=[^ ]+ .*total_s=[0-9.]+' "$LOG" | grep -oE 'total_s=[0-9.]+' | cut -d= -f2 | paste -sd, -)
SRC=PHASE.total_s
if [ -z "$FWD" ]; then FWD=$(grep -oE 'Model forward time: [0-9.]+s' "$LOG" | grep -oE '[0-9.]+' | paste -sd, -); SRC="Model forward time"; fi
WARM=$(echo "$FWD" | tr ',' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END {if (NR == 0) print "NA"; else if (NR % 2) printf "%.3f\n", a[(NR + 1) / 2]; else printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2}')
FEAT=$(grep -oE 'TEMPLATES FEATURISED query=[^ ]+ real_slots=[0-9]+/[0-9]+' "$LOG")
NFEAT=$(echo "$FEAT" | grep -c real_slots)
TREAL=$(echo "$FEAT" | grep -oE 'real_slots=[0-9]+' | cut -d= -f2 | sort -n | head -1)
[ "$NFEAT" -eq "$COPIES" ] || TREAL=0            # an item without its FEATURISED line counts as no real template
MSA=$(grep -m1 -oE '(^|[ ,\[])msa\[1, 1, [0-9]+, ' "$LOG" | grep -oE '[0-9]+, $' | tr -d ', ')
{
  echo "BENCH fwd source: $SRC ($(echo "$FWD" | tr ',' '\n' | grep -c .) forwards, expected $COPIES)"
  echo "BENCH templates: FEATURISED lines=$NFEAT/$COPIES min real_slots=${TREAL:-0}; per-chain: $(grep -oE 'TEMPLATES chain=[^ ]+ real=[0-9]+/[0-9]+ dropped=[0-9]+' "$LOG" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  echo "BENCH templates added to OF3's cache (upstream template log): $(cat "$R"/tlogs/*.log 2>/dev/null | grep -E 'added to cache|Selected chain|does not pass|could not|no aligned|missing' | tr '\n' ';')"
  echo "BENCH drops: $(grep -E 'TEMPLATES DROPPED|TEMPLATE DROPPED|TEMPLATE INPUT DROPPED|ALL_DUMMY' "$LOG" | head -5 | tr '\n' ';')"
  echo "BENCH distinct census: $(grep -oE 'templ distinct census: .*' "$LOG" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  echo "BENCH msa feature: $(grep -m1 -oE 'msa\[1, 1, [0-9]+, [0-9]+, [0-9]+\]' "$LOG")"
} >> "$LOG"
LINE="KIT $KIT n=$N exit=$RC fwd=[$FWD] warm_median=$WARM templates_real=${TREAL:-0}/4 msa_rows=${MSA:-?}"
echo "$LINE" >> "$LOG"
echo "$LINE"
