#!/bin/bash
# The 5I28 benchmark, RoseTTAFold3 Anthropic kit, ALL n in ONE kit process: inputs=<dir> holding every size's items (n01_c0..c3, n02_c0..c2, n04_, n08_, n12_c0..c1, n16_c0..c1;
# same copies per size as run.sh, same fold flags, same fwdtimer.py FWDTIME span). Per-call size is read from the REALINPUT n_token (=128*n), so the file order does not matter.
# Result lines per n: logs/all.results; done markers $DONE/kit_rf3_x<n>.
set -uo pipefail
D=$BENCH_DIR/rf3; K=$KITS_DIR; KD=$K/rosettafold3_cu130; MODE=${MODE:-fast}
DONE=${DONE:-$BENCH_DIR/done_final}; KIT=rosettafold3; [ "$MODE" = fast ] || KIT=rosettafold3:$MODE
R=$D/runs/all; LOG=$D/logs/all.log; MSA=$D/inputs/msa/5i28.a3m
rm -rf "$R"; mkdir -p "$R/out" "$R/many" "$D/logs" "$D/tmp" "$D/cache/jit" "$DONE"
for sf in 1:4 2:3 4:3 8:3 12:2 16:2; do N=${sf%:*}; C=${sf#*:}; IN=$D/inputs/x$N; PDB=$IN/tpl_x$N.pdb
  LO=$(grep -oE '"res_range": \[[0-9]+, [0-9]+\]' "$IN/tpl_info.json" | grep -oE '[0-9]+' | head -1); HI=$(grep -oE '"res_range": \[[0-9]+, [0-9]+\]' "$IN/tpl_info.json" | grep -oE '[0-9]+' | tail -1)
  MSAS=""; SEL=""; for ((c = 0; c < N; c++)); do ch=$(printf "\\x$(printf %x $((65 + c)))"); MSAS+="${MSAS:+, }\"$ch\": \"$MSA\""; SEL+="${SEL:+, }\"$ch/*/$LO-$HI\""; done
  for ((k = 0; k < C; k++)); do echo "[{\"name\": \"n$(printf %02d $N)_c$k\", \"components\": [{\"path\": \"$PDB\"}], \"msa_paths\": {$MSAS}, \"template_selection\": [$SEL]}]" > "$R/many/n$(printf %02d $N)_c$k.json"; done
done
T13=$FF_SCRATCH/cuda13tk/.pixi/envs/default
export CC=$T13/bin/x86_64-conda-linux-gnu-gcc CXX=$T13/bin/x86_64-conda-linux-gnu-g++ CUDA_HOME=$T13 CU130_EXTRA_LD=$T13/lib \
       TRITON_LIBCUDA_PATH=$FF_SCRATCH/cudacompat13/usr/local/cuda-13.0/compat
export PATH=$T13/bin:$PATH
export XDG_CACHE_HOME=$D/cache TRITON_CACHE_DIR=$D/cache/triton TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor TORCH_EXTENSIONS_DIR=$D/cache/torch_ext \
       MODEL_OPT_JIT_ROOT=$D/cache/jit HF_HOME=$D/cache/hf TORCH_HOME=$D/cache/torch TMPDIR=$D/tmp PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
CK=$K/weights/rosettafold3/rf3_foundry_01_24_latest_remapped.ckpt
export FOUNDRY_CHECKPOINT_DIRS=$K/weights/rosettafold3 ROSETTAFOLD3_OPT_CKPT=$CK MODEL_OPT=$KD
unset PYTHONPATH
FOLD="ckpt_path=$CK inputs=$R/many out_dir=$R/out n_recycles=4 num_steps=200 diffusion_batch_size=5 early_stopping_plddt_threshold=0 seed=0"
{ echo "BENCH 5I28 kit=rosettafold3_cu130 mode=$MODE ALL-n host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "BENCH gpu: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
  echo "BENCH input: $R/many ($(ls $R/many | wc -l) items; msa $MSA, $(grep -c '^>' "$MSA") a3m rows; template = threaded tpl_x<n>.pdb, 5syd_A on 88 aligned positions)"; } > "$LOG"
cd "$KD"
timeout --kill-after=60 14400 "$D/cu130kit.sh" bash -c "set -e; cd $KD; source configs/h100.env; export ROSETTAFOLD3_OPT=$MODE; exec $KD/opt/venv/bin/python $D/fwdtimer.py rf3.model.RF3.RF3WithConfidence rf3.cli:app fold $FOLD" >> "$LOG" 2>&1
RC=$?
: > $D/logs/all.results
for n in 1 2 4 8 12 16; do
  EXP=$([ $n = 1 ] && echo 4 || { [ $n -ge 12 ] && echo 2 || echo 3; })
  # calls whose REALINPUT n_token == 128*n, with the FWDTIME of the same call index
  CALLS=$(grep -aE "^REALINPUT call=[0-9]+ .* n_token=$((128 * n)) " "$LOG" | grep -oE 'call=[0-9]+' | cut -d= -f2)
  FWD=""; NOK=0; for c in $CALLS; do t=$(grep -aE "^FWDTIME $c " "$LOG" | awk '{print $3}' | head -1); [ -n "$t" ] && FWD+="${FWD:+,}$t"
    grep -aqE "^REALINPUT call=$c .* templated_tokens=$((88 * n)) template_pairs=$((n * 88 * 88)) " "$LOG" && NOK=$((NOK + 1)); done
  NFW=$(echo "$FWD" | tr ',' '\n' | grep -c .)
  WARM=$(echo "$FWD" | tr ',' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END {if (NR == 0) print "NA"; else if (NR % 2) printf "%.3f\n", a[(NR + 1) / 2]; else printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2}')
  TREAL=0; [ "$NFW" -ge "$EXP" ] && [ "$NOK" -eq "$NFW" ] && TREAL=1
  S=ok; [ "$NFW" -lt "$EXP" ] && { if grep -aqiE "out of memory|OutOfMemory" "$LOG"; then S=oom; else S=failed; fi; }
  MROWS=$(grep -aoE '^MSAFULL rows=[0-9]+ tokens='$((128 * n))' ' "$LOG" | grep -oE 'rows=[0-9]+' | cut -d= -f2 | sort -u | paste -sd/ -)
  echo "KIT $KIT n=$n exit=$([ $S = ok ] && echo 0 || echo $S) fwd=[$FWD] warm_median=$WARM templates_real=$TREAL/1 msa_rows=${MROWS:-?} process_rc=$RC" | tee -a $D/logs/all.results
  echo "job=${SLURM_JOB_ID:-} host=$(hostname) single-process KIT $KIT n=$n status=$S fwd=[$FWD] warm_median=$WARM" > $DONE/kit_rf3_x$n
done
