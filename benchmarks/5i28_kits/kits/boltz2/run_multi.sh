#!/bin/bash
# usage: run_multi.sh  -- the 5I28 benchmark on the Boltz-2 Anthropic kit, ALL n in ONE kit process: the input dir holds the copies of every size in ascending n
#   (names a00 .. a16: n=1 x4 copies, n=2,4,8 x3, n=12,16 x2; the kit runs them one after the other, the first copy of each size is cold for that shape;
#   same F as the per-cell run.sh). Same kit command / flags / timer (PHASE fwd_s = Boltz2.forward, CUDA-synchronized) as run.sh. Result lines per n in logs/all.results.
MODE=${MODE:-fast}; H=$BENCH_DIR/boltz2; KIT=$KITS_DIR/boltz2/boltz2; DONE=${DONE:-$BENCH_DIR/done_final}
D=$H/out/all; L=$H/logs/all.log
rm -rf "$D"; mkdir -p "$D/in" "$H/logs" "$DONE"
SIZES="1:4 2:3 4:3 8:3 12:2 16:2"; idx=0; MAP=""
for sf in $SIZES; do n=${sf%:*}; f=${sf#*:}; for ((k=0;k<f;k++)); do nm=$(printf "a%02d" $idx); cp "$H/inputs/5i28_x$n.yaml" "$D/in/$nm.yaml"; MAP="$MAP $nm=$n"; idx=$((idx+1)); done; done
cat > "$D/inner.sh" <<EOT
source $H/scripts/env_in.sh
cd $D
echo "[bench5i28] boltz2 ALL-n mode=$MODE inputs=$idx host=\$(hostname) CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset} \$(date -Is)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
bash $KIT/run.sh pred --config h100 --mode $MODE --input $D/in --out_dir $D/out --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5 --seed 0 --output_format mmcif --override
rc=\$?
echo "[bench5i28] kit rc=\$rc"
python -I $H/scripts/msa_npz.py $D/out
exit \$rc
EOT
CU130_EXTRA_LD=$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib "$H/scripts/cu130run.sh" bash "$D/inner.sh" > "$L" 2>&1
RC=$?
W=$D/out/pred_worker.log
if [ -f "$W" ]; then { echo; echo "===== $W (appended by run_multi.sh) ====="; cat "$W"; } >> "$L"; fi
: > $H/logs/all.results
for n in 1 2 4 8 12 16; do
  NMS=$(for m in $MAP; do [ "${m#*=}" = "$n" ] && echo -n "${m%=*}|"; done | sed 's/|$//')
  FWD=$(grep -aE "^PHASE item=($NMS) " "$L" | grep -oE 'fwd_s=[0-9.]+' | cut -d= -f2 | tr '\n' ' ' | sed 's/ $//')
  WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END{if(NR) printf "%.3f", (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; else printf "NA"}')
  TREC=$(grep -aoE "TEMPLATES record=($NMS) declared=[0-9]+ names=[0-9]+ slots=[0-9]+ live_slots=[0-9]+" "$L" | sort -u | awk '{for(i=1;i<=NF;i++){split($i,a,"="); v[a[1]]=a[2]} m=(v["names"]<v["live_slots"])?v["names"]:v["live_slots"]; if(min==""||m<min)min=m} END{print (min=="")?0:min}')
  TNAM=0; for t in 5syd_A 6kol_A 5yt7_A 6tfd_A; do grep -aqE "TEMPLATE record=($NMS) template=$t chain=[A-Z]+ aligned_residues_resolved=[1-9]" "$L" && TNAM=$((TNAM+1)); done
  TREAL=$(( TREC < TNAM ? TREC : TNAM ))
  NFW=$(echo $FWD | wc -w); NEXP=$(echo $NMS | tr '|' '\n' | wc -l); S=ok; [ "$NFW" -lt "$NEXP" ] && { if grep -aqiE "out of memory|OutOfMemory" "$L"; then S=oom; else S=failed; fi; }
  echo "KIT boltz2 n=$n exit=$([ $S = ok ] && echo 0 || echo $S) fwd=[$FWD] warm_median=$WARM templates_real=$TREAL/4 msa_rows=$(grep -aoE '^MSA_ROWS file=\S+ sequences=[0-9]+' "$L" | grep -oE 'sequences=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ',' | sed 's/,$//') process_rc=$RC" | tee -a $H/logs/all.results
  echo "job=${SLURM_JOB_ID:-} host=$(hostname) single-process KIT boltz2 n=$n status=$S fwd=[$FWD] warm_median=$WARM" > $DONE/kit_boltz2_x$n
done
