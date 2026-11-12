#!/bin/bash
# usage: run.sh N        (N = 1 2 4 8 12 16; on a GPU node, CUDA_VISIBLE_DEVICES set by the caller; host shell, NOT inside the container)
# The 5I28 benchmark (miniworld-engine benchmarks/5i28.md) on the Boltz-2 Anthropic kit, pinned cu130 stack (container as anthropic-kits/cu130run.sh):
#   N identical 5I28 chains, real MSA (5i28_unpaired.a3m, all rows) on every chain, 4 distinct templates 5syd_A 6kol_A 5yt7_A 6tfd_A (mmCIF + chain A,
#   every query chain), recycling_steps 3 (= 4 trunk passes), 200 steps, 5 samples, kit mode $MODE (default fast), seed 0.
#   F forwards in one process = F copies of the input (a0..a<F-1>): F = 4 (N=1), 3 (N<=8), 2 (N>=12); FWDS=<k> overrides.
# Writes logs/x<N>.log (the kit's stdout+stderr, then pred_worker.log appended) and prints ONE line:
#   KIT boltz2 n=<N> exit=<rc> fwd=[<PHASE fwd_s per forward, cold first>] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
N=$1; MODE=${MODE:-fast}
case "$N" in 1|2|4|8|12|16) ;; *) echo "usage: run.sh N   (N = 1 2 4 8 12 16)" >&2; exit 2 ;; esac
H=$BENCH_DIR/boltz2; KIT=$KITS_DIR/boltz2/boltz2
if [ -n "${FWDS:-}" ]; then F=$FWDS; elif [ "$N" -le 1 ]; then F=4; elif [ "$N" -le 8 ]; then F=3; else F=2; fi
D=$H/out/x$N; L=$H/logs/x$N.log
rm -rf "$D"; mkdir -p "$D/in" "$H/logs"
for ((k=0;k<F;k++)); do cp "$H/inputs/5i28_x$N.yaml" "$D/in/a$k.yaml"; done
cat > "$D/inner.sh" <<EOT
source $H/scripts/env_in.sh
cd $D
echo "[bench5i28] boltz2 n=$N mode=$MODE forwards=$F host=\$(hostname) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset} \$(date -Is)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
bash $KIT/run.sh pred --config h100 --mode $MODE --input $D/in --out_dir $D/out --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5 --seed 0 --output_format mmcif --override
rc=\$?
echo "[bench5i28] kit rc=\$rc"
python -I $H/scripts/msa_npz.py $D/out
exit \$rc
EOT
CU130_EXTRA_LD=$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib "$H/scripts/cu130run.sh" bash "$D/inner.sh" > "$L" 2>&1
RC=$?
# forward seconds: the kit's PHASE line per input, fwd_s = Boltz2.forward, CUDA-synchronized (audit 2026-10-29: same span as FoldForge)
FWD=$(grep -E '^PHASE item=a[0-9]+ ' "$L" | grep -oE 'fwd_s=[0-9.]+' | cut -d= -f2 | tr '\n' ' ' | sed 's/ $//')
WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END{if(NR) printf "%.3f", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; else printf "NA"}')
# templates: the kit's model-entry census per input ([boltz2-opt] TEMPLATES record=.. names=<distinct> live_slots=<j> real=<j>/<T>), min over inputs of
# min(names, live_slots); capped by the number of the 4 expected names that the featurizer census (pred_worker.log TEMPLATE lines) shows with resolved residues
TREC=$(grep -oE 'TEMPLATES record=a[0-9]+ declared=[0-9]+ names=[0-9]+ slots=[0-9]+ live_slots=[0-9]+' "$L" | sort -u | \
       awk '{for(i=1;i<=NF;i++){split($i,a,"="); v[a[1]]=a[2]} m=(v["names"]<v["live_slots"])?v["names"]:v["live_slots"]; if(min==""||m<min)min=m} END{print (min=="")?0:min}')
W=$D/out/pred_worker.log
TNAM=0; for t in 5syd_A 6kol_A 5yt7_A 6tfd_A; do grep -qE "TEMPLATE record=a[0-9]+ template=$t chain=[A-Z]+ aligned_residues_resolved=[1-9]" "$W" 2>/dev/null && TNAM=$((TNAM+1)); done
TREAL=$(( TREC < TNAM ? TREC : TNAM ))
MSA=$(grep -oE '^MSA_ROWS file=\S+ sequences=[0-9]+' "$L" | grep -oE 'sequences=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ',' | sed 's/,$//')
if [ -f "$W" ]; then { echo; echo "===== $W (appended by run.sh) ====="; cat "$W"; } >> "$L"; fi
S="KIT boltz2 n=$N exit=$RC fwd=[$FWD] warm_median=$WARM templates_real=$TREAL/4 msa_rows=${MSA:-NA}"
{ echo; echo "[bench5i28] census: model-entry real=min(names,live_slots)=$TREC, expected names resolved in featurizer census=$TNAM/4"; echo "$S"; } >> "$L"
echo "$S"
