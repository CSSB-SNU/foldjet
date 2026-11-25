#!/bin/bash
# The 5I28 benchmark, RoseTTAFold3 Anthropic kit ($KITS_DIR/rosettafold3_cu130, used read-only), ONE n on the GPU the
# caller gives (CUDA_VISIBLE_DEVICES). Real MSA (msa_paths, every chain) + ONE template: RF3 takes one coordinate set per chain through its
# input-structure template track, so the input structure is a threaded backbone (query sequence, 5syd_A coordinates on its 88 aligned query
# positions 41-128, occupancy 0 elsewhere; mkthread.py) and template_selection = that residue range of every chain. n_recycles=4,
# num_steps=200, diffusion_batch_size=5, seed=0, kit precision of the mode. One process folds COPIES identical items = COPIES forwards.
# usage: run.sh N        env: MODE=fast|exact|off (default fast = the kit mode compared so far), COPIES (default 4 at n=1, else 3)
# Full log: logs/x<N>.log. Stdout: ONE line
#   KIT rosettafold3 n=<N> exit=<rc> fwd=[<s>,...] warm_median=<s> templates_real=<k>/1 msa_rows=<r>
# fwd = `FWDTIME` of fwdtimer.py: RF3WithConfidence.forward between torch.cuda.synchronize() calls (audit: same span as FoldJet), item order.
# templates_real = 1/1 when EVERY forward's input (REALINPUT) has templated_tokens = 88*N and template_pairs = N*88*88 (5syd_A on every
#   chain, intra-chain pairs only), else 0/1.
# msa_rows = the full MSA depth handed to atomworks' FeaturizeMSALikeAF3 (MSAFULL line); RF3 then feeds n_msa rows per recycle (msa_stack).
set -uo pipefail
N=${1:?usage: run.sh N}
D=$BENCH_DIR/rf3
K=$KITS_DIR
KD=$K/rosettafold3_cu130
MODE=${MODE:-fast}
COPIES=${COPIES:-$([ "$N" = 1 ] && echo 4 || echo 3)}
KIT=rosettafold3; [ "$MODE" = fast ] || KIT=rosettafold3:$MODE
R=$D/runs/x$N; LOG=$D/logs/x$N.log
IN=$D/inputs/x$N; PDB=$IN/tpl_x$N.pdb; MSA=$D/inputs/msa/5i28.a3m
[ -f "$PDB" ] && [ -f "$IN/tpl_info.json" ] || { echo "KIT $KIT n=$N exit=2 fwd=[] warm_median=NA templates_real=0/1 msa_rows=? (missing $PDB)"; exit 2; }
LO=$(grep -oE '"res_range": \[[0-9]+, [0-9]+\]' "$IN/tpl_info.json" | grep -oE '[0-9]+' | head -1)
HI=$(grep -oE '"res_range": \[[0-9]+, [0-9]+\]' "$IN/tpl_info.json" | grep -oE '[0-9]+' | tail -1)
NAL=$(grep -oE '"complete_backbone": [0-9]+' "$IN/tpl_info.json" | grep -oE '[0-9]+$')
rm -rf "$R"; mkdir -p "$R/out" "$R/many" "$D/logs" "$D/tmp" "$D/cache/jit"
CH=(); for ((c = 0; c < N; c++)); do CH+=("$(printf "\\x$(printf %x $((65 + c)))")"); done
MSAS=""; SEL=""; for c in "${CH[@]}"; do MSAS+="${MSAS:+, }\"$c\": \"$MSA\""; SEL+="${SEL:+, }\"$c/*/$LO-$HI\""; done
for ((k = 0; k < COPIES; k++)); do
  echo "[{\"name\": \"c$k\", \"components\": [{\"path\": \"$PDB\"}], \"msa_paths\": {$MSAS}, \"template_selection\": [$SEL]}]" > "$R/many/c$k.json"
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
{
  echo "BENCH 5I28 kit=rosettafold3_cu130 mode=$MODE n=$N tokens=$((128 * N)) copies=$COPIES host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "BENCH gpu: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tr '\n' ';')"
  echo "BENCH input: $R/many/c0.json (msa $MSA, $(grep -c '^>' "$MSA") a3m rows, every chain; template = threaded $PDB: $(cat "$IN/tpl_info.json"); template_selection <chain>/*/$LO-$HI)"
} > "$LOG"
cd "$KD"
if [ "$MODE" = off ]; then
  timeout --kill-after=60 7200 "$D/cu130kit.sh" "$KD/venv/bin/python" "$D/fwdtimer.py" rf3.model.RF3.RF3WithConfidence rf3.cli:app fold $FOLD >> "$LOG" 2>&1
else
  timeout --kill-after=60 7200 "$D/cu130kit.sh" bash -c "set -e; cd $KD; source configs/h100.env; export ROSETTAFOLD3_OPT=$MODE; exec $KD/opt/venv/bin/python $D/fwdtimer.py rf3.model.RF3.RF3WithConfidence rf3.cli:app fold $FOLD" >> "$LOG" 2>&1
fi
RC=$?
# ---- evidence from the run's own log ----
FWD=$(grep -oE '^FWDTIME [0-9]+ [0-9.]+' "$LOG" | awk '{print $3}' | paste -sd, -)
WARM=$(echo "$FWD" | tr ',' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END {if (NR == 0) print "NA"; else if (NR % 2) printf "%.3f\n", a[(NR + 1) / 2]; else printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2}')
NRI=$(grep -c '^REALINPUT call=' "$LOG")
NOK=$(grep -cE "^REALINPUT call=[0-9]+ .* templated_tokens=$((NAL * N)) template_pairs=$((N * NAL * NAL)) " "$LOG")
NFWD=$(echo "$FWD" | tr ',' '\n' | grep -c .)
TREAL=0; [ "$NFWD" -gt 0 ] && [ "$NRI" -eq "$NFWD" ] && [ "$NOK" -eq "$NRI" ] && TREAL=1
MROWS=$(grep -oE '^MSAFULL rows=[0-9]+' "$LOG" | cut -d= -f2 | sort -u | paste -sd/ -)
{
  echo "BENCH fwd source: FWDTIME (RF3WithConfidence.forward, synchronised) $NFWD forwards, expected $COPIES"
  echo "BENCH templates: REALINPUT lines=$NRI, with templated_tokens=$((NAL * N)) and template_pairs=$((N * NAL * NAL)): $NOK; kit: $(grep -oE 'TEMPLATE selection=[0-9]+ atoms_selected=[0-9]+/[0-9]+' "$LOG" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  echo "BENCH msa: full depth (MSAFULL) $(grep -E '^MSAFULL' "$LOG" | sort | uniq -c | tr -s ' ' | tr '\n' ';') per-recycle stack: $(grep -oE 'msa_stack=\([0-9, ]+\)' "$LOG" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
} >> "$LOG"
LINE="KIT $KIT n=$N exit=$RC fwd=[$FWD] warm_median=$WARM templates_real=$TREAL/1 msa_rows=${MROWS:-?}"
echo "$LINE" >> "$LOG"
echo "$LINE"
