#!/bin/bash
# usage: run.sh N        (N = 1 2 4 8 12 16; on a GPU node, CUDA_VISIBLE_DEVICES set by the caller; host shell, NOT inside the container)
# The 5I28 benchmark (miniworld-engine benchmarks/5i28.md) on the Chai-1 Anthropic kit, pinned cu130 stack (container as anthropic-kits/cu130run.sh):
#   N identical 5I28 chains (inputs/5i28_x<N>.fasta), real MSA (inputs/msa/*.aligned.pqt = 5i28_unpaired.a3m, all rows) via --msa-directory,
#   4 distinct templates 5SYD_A 6KOL_A 5YT7_A 6TFD_A per chain via --template-hits-path inputs/templates_n<N>.m8 (+ inputs/cifs, offline shim),
#   --num-trunk-recycles 3 (= 4 trunk passes), 200 steps, 5 samples, kit mode $MODE (default fast), --seed 0.
#   F forwards in one process = --num-trunk-samples F (seeds 0..F-1): F = 4 (N=1), 3 (N<=8), 2 (N>=12); FWDS=<k> overrides.
# Timing: the shim's FWDSPAN fwd_s (first load_exported .. last confidence-head forward, CUDA events), NOT the kit's FORWARD line (that one
# also collates the features and ranks + writes the CIFs; it stays in the log). See README.md.
# Writes logs/x<N>.log (the kit's stdout+stderr) and prints ONE line:
#   KIT chai1 n=<N> exit=<rc> fwd=[<FWDSPAN fwd_s per forward, cold first>] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
N=$1; MODE=${MODE:-fast}
case "$N" in 1|2|4|8|12|16) ;; *) echo "usage: run.sh N   (N = 1 2 4 8 12 16)" >&2; exit 2 ;; esac
H=$BENCH_DIR/chai1; KIT=$KITS_DIR/chai1/chai1
if [ -n "${FWDS:-}" ]; then F=$FWDS; elif [ "$N" -le 1 ]; then F=4; elif [ "$N" -le 8 ]; then F=3; else F=2; fi
D=$H/out/x$N; L=$H/logs/x$N.log
rm -rf "$D"; mkdir -p "$D/in" "$D/out" "$H/logs"
cp "$H/inputs/5i28_x$N.fasta" "$D/in/"; cp "$H/inputs/templates_n$N.m8" "$D/in/templates.m8"
cat > "$D/inner.sh" <<EOT
source $H/scripts/env_in.sh
cd $D
echo "[bench5i28] chai1 n=$N mode=$MODE forwards=$F host=\$(hostname) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset} \$(date -Is)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader; which kalign
bash $KIT/run.sh pred --config h100 --mode $MODE --seed 0 --input $D/in/5i28_x$N.fasta --out_dir $D/out --msa-directory $H/inputs/msa \
  --template-hits-path $D/in/templates.m8 --num-trunk-recycles 3 --num-diffn-timesteps 200 --num-diffn-samples 5 --num-trunk-samples $F
rc=\$?
echo "[bench5i28] kit rc=\$rc"
exit \$rc
EOT
CU130_EXTRA_LD=$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib "$H/scripts/cu130run.sh" bash "$D/inner.sh" > "$L" 2>&1
RC=$?
FWD=$(grep -oE '^\[bench5i28\] FWDSPAN fold=[0-9]+ .* fwd_s=[0-9.NA]+' "$L" | grep -oE 'fwd_s=[0-9.NA]+' | cut -d= -f2 | tr '\n' ' ' | sed 's/ $//')
WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | grep -E '^[0-9.]+$' | sort -g | awk '{a[NR]=$1} END{if(NR) printf "%.3f", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; else printf "NA"}')
KFWD=$(grep -oE 'FORWARD item=\S+ .*forward_s=[0-9.]+' "$L" | grep -oE 'forward_s=[0-9.]+' | cut -d= -f2 | tr '\n' ' ' | sed 's/ $//')
# templates: chai_lab's own "Loaded k templates: [...]" (one line per chain per feature-context build): the number of the 4 expected names
# listed, min over lines (0 if no line, or fewer lines than chains); capped by the CONTEXT line's nonnull_templates (the merged TemplateContext)
NL=$(grep -cE 'Loaded [0-9]+ templates: \[' "$L")
TL=$(grep -oE 'Loaded [0-9]+ templates: \[[^]]*\]' "$L" | while read -r l; do c=0; for t in 5SYD 6KOL 5YT7 6TFD; do grep -q "'$t|A'" <<<"$l" && c=$((c+1)); done; echo $c; done | sort -n | head -1)
[ -z "$TL" ] || [ "$NL" -lt "$N" ] && TL=0
TC=$(grep -oE '^\[bench5i28\] CONTEXT fold=[0-9]+ .* nonnull_templates=[0-9]+' "$L" | grep -oE 'nonnull_templates=[0-9]+' | cut -d= -f2 | sort -n | head -1)
TREAL=$TL; [ -n "$TC" ] && [ "$TC" -lt "$TREAL" ] && TREAL=$TC
MSA=$(grep -oE 'MSA found for sequence: [A-Z]+, msa\.depth=[0-9]+' "$L" | grep -oE 'depth=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ',' | sed 's/,$//')
[ -z "$MSA" ] && MSA=$(grep -oE '^\[bench5i28\] CONTEXT .* msa_depth=[0-9]+' "$L" | grep -oE 'msa_depth=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ',' | sed 's/,$//')
S="KIT chai1 n=$N exit=$RC fwd=[$FWD] warm_median=$WARM templates_real=$TREAL/4 msa_rows=${MSA:-NA}"
{ echo; echo "[bench5i28] census: Loaded-lines=$NL (chains $N) min expected names listed=${TL} context nonnull_templates(min)=${TC:-NA}; kit FORWARD (whole run_folding_on_context, not the benchmark span)=[$KFWD]"
  grep -E 'PAIRTRACK GATE REFUSED|seed_folds_ok|\[chai1-opt\] EXIT' "$L" | cut -c1-300 | sed 's/^/[bench5i28] kit verdict: /'
  echo "$S"; } >> "$L"
echo "$S"
