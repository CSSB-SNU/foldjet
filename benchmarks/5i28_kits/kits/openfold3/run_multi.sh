#!/bin/bash
# The 5I28 benchmark, OpenFold3 Anthropic kit, ALL n in ONE kit process: one query JSON with every size's identical queries in ascending n
# (queries n01_c0..n01_c3, n02_c0..c2, n04_.., n08_.., n12_c0..c1, n16_c0..c1: the same copies per size as the per-cell run.sh); the first copy of each size is cold
# for that shape. Same kit command, flags and timer (PHASE item=.. total_s) as run.sh. Result lines per n: logs/all.results; done markers $DONE/kit_openfold3_x<n>.
set -uo pipefail
D=$BENCH_DIR/openfold3; K=$KITS_DIR; KD=$K/openfold3_ob0; MODE=${MODE:-fast}
DONE=${DONE:-$BENCH_DIR/done_final}; KIT=openfold3; [ "$MODE" = fast ] || KIT=openfold3:$MODE
R=$D/runs/all; LOG=$D/logs/all.log
rm -rf "$R"; mkdir -p "$R/out" "$R/tlogs" "$D/logs" "$D/tmp" "$D/cache/openfold_cache" "$DONE"
SEQ=$(awk '!/^>/ {print; exit}' "$D/inputs/msa/A/colabfold_main.a3m"); [ ${#SEQ} -eq 128 ] || { echo "query length ${#SEQ} != 128" >&2; exit 1; }
TPL=(5syd:A 6kol:A 5yt7:A 6tfd:A); paths=""; chains=""
for t in "${TPL[@]}"; do paths+="${paths:+, }\"$D/inputs/template_cifs/${t%%:*}.cif\""; chains+="${chains:+, }\"${t##*:}\""; done
{ printf '{"seeds": [0], "queries": {'; first=1
  for sf in 1:4 2:3 4:3 8:3 12:2 16:2; do n=${sf%:*}; c=${sf#*:}; ids=""
    for ((k = 0; k < n; k++)); do ids+="${ids:+, }\"$(printf "\\x$(printf %x $((65 + k)))")\""; done
    for ((k = 0; k < c; k++)); do printf '%s"n%02d_c%d": {"chains": [{"molecule_type": "protein", "chain_ids": [%s], "sequence": "%s", "main_msa_file_paths": "%s", "template_cif_paths": [%s], "template_cif_chain_ids": [%s]}]}' "$([ $first = 0 ] && echo ', ')" $n $k "$ids" "$SEQ" "$D/inputs/msa/A" "$paths" "$chains"; first=0; done
  done; printf '}}\n'; } > "$R/many.json"
cat > "$R/runner.yml" <<EOF2
template_preprocessor_settings:
  structure_directory: $D/inputs/template_cifs
  structure_file_format: cif
  fetch_missing_structures: false
  output_directory: $R/template_data
  create_logs: true
  log_directory: $R/tlogs
EOF2
T13=$FF_SCRATCH/cuda13tk/.pixi/envs/default
export CC=$T13/bin/x86_64-conda-linux-gnu-gcc CXX=$T13/bin/x86_64-conda-linux-gnu-g++ CUDA_HOME=$T13 CU130_EXTRA_LD=$T13/lib \
       TRITON_LIBCUDA_PATH=$FF_SCRATCH/cudacompat13/usr/local/cuda-13.0/compat
export PATH=$KD/venv/bin:$T13/bin:$PATH
export XDG_CACHE_HOME=$D/cache TRITON_CACHE_DIR=$D/cache/triton TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TORCH_EXTENSIONS_DIR=$D/cache/torch_ext \
       MODEL_OPT_JIT_ROOT=$D/cache/jit HF_HOME=$D/cache/hf TORCH_HOME=$D/cache/torch TMPDIR=$D/tmp PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENFOLD3_OB0_CKPT=$K/weights/openfold3/of3-ob-2025-06-30-174k.pt OPENFOLD_CACHE=$D/cache/openfold_cache
unset PYTHONPATH
{ echo "BENCH 5I28 kit=openfold3_ob0 mode=$MODE ALL-n host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "BENCH gpu: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
  echo "BENCH input: $R/many.json ($(grep -o '"n[0-9]*_c[0-9]*"' "$R/many.json" | wc -l) queries; msa $D/inputs/msa/A/colabfold_main.a3m, $(grep -c '^>' "$D/inputs/msa/A/colabfold_main.a3m") a3m rows; templates 5syd_A 6kol_A 5yt7_A 6tfd_A CIF-direct)"; } > "$LOG"
cd "$KD"
timeout --kill-after=60 14400 "$D/cu130kit.sh" bash run.sh pred --config h100 --mode "$MODE" --query-json "$R/many.json" --output-dir "$R/out" \
  --use-msa-server false --num-diffusion-samples 5 --runner-yaml "$R/runner.yml" >> "$LOG" 2>&1
RC=$?
: > $D/logs/all.results
for n in 1 2 4 8 12 16; do
  P=$(printf 'n%02d_c' $n)
  FWD=$(grep -aE "PHASE item=${P}[0-9]+ .*total_s=[0-9.]+" "$LOG" | grep -oE 'total_s=[0-9.]+' | cut -d= -f2 | paste -sd, -)
  WARM=$(echo "$FWD" | tr ',' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END {if (NR == 0) print "NA"; else if (NR % 2) printf "%.3f\n", a[(NR + 1) / 2]; else printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2}')
  EXP=$([ $n = 1 ] && echo 4 || { [ $n -ge 12 ] && echo 2 || echo 3; })
  FEAT=$(grep -aoE "TEMPLATES FEATURISED query=${P}[0-9]+ real_slots=[0-9]+/[0-9]+" "$LOG"); NFEAT=$(echo "$FEAT" | grep -c real_slots)
  TREAL=$(echo "$FEAT" | grep -oE 'real_slots=[0-9]+' | cut -d= -f2 | sort -n | head -1); [ "$NFEAT" -eq "$EXP" ] || TREAL=0
  NFW=$(echo "$FWD" | tr ',' '\n' | grep -c .); S=ok; [ "$NFW" -lt "$EXP" ] && { if grep -aqiE "out of memory|OutOfMemory" "$LOG"; then S=oom; else S=failed; fi; }
  MSA=$(grep -a -m1 -oE '(^|[ ,\[])msa\[1, 1, [0-9]+, ' "$LOG" | grep -oE '[0-9]+, $' | tr -d ', ')
  echo "KIT $KIT n=$n exit=$([ $S = ok ] && echo 0 || echo $S) fwd=[$FWD] warm_median=$WARM templates_real=${TREAL:-0}/4 msa_rows=${MSA:-?} process_rc=$RC" | tee -a $D/logs/all.results
  echo "job=${SLURM_JOB_ID:-} host=$(hostname) single-process KIT $KIT n=$n status=$S fwd=[$FWD] warm_median=$WARM" > $DONE/kit_openfold3_x$n
done
