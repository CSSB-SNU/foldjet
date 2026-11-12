#!/bin/bash
# usage: run_multi.sh  -- ESMFold2 kit (fast, variant full_msa), the 5I28 benchmark, ALL n in one process (esm_fwd_multi.py); results to logs/all.log, all.results, done markers
K=$KITS_DIR; W=$K/esmfold2; D=$BENCH_DIR/esmfold2; DONE=${DONE:-$BENCH_DIR/done_final}
mkdir -p $D/tmp $D/logs $DONE
cat > $D/tmp/inner_all.sh <<EOT
export XDG_CACHE_HOME=$D/cache UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip TMPDIR=$D/tmp
export TRITON_CACHE_DIR=$D/cache/triton TORCH_EXTENSIONS_DIR=$D/cache/torch_ext TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor
export MODEL_OPT_JIT_ROOT=$D/jitroot HF_HOME=$W/weights HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt PYTHONHASHSEED=0 NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export CC=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/gcc CXX=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/g++
. $W/venv130/bin/activate
cd $W/esmfold2
source configs/h100.env
export ESMFOLD2_OPT=fast ESMFOLD2_VARIANT=full_msa FWD_CALLS=4 SIZES="1 2 4 8 12 16"
python $D/esm_fwd_multi.py
EOT
L=$D/logs/all.log; : > $L
$K/cu130run.sh bash $D/tmp/inner_all.sh > $L 2>&1; RC=$?; echo "CONTAINER_RC=$RC" >> $L
: > $D/logs/all.results
for n in 1 2 4 8 12 16; do
  FWD=$(grep -aoE "^FWDTIME n=$n [0-9]+ [0-9.]+" $L | awk '{print $4}' | tr '\n' ' ')
  WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END{if(NR) print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  NFW=$(echo $FWD | wc -w); S=ok; [ "$NFW" -lt 4 ] && { if grep -aq "^FWDOOM $n " $L; then S=oom; else S=failed; fi; }
  echo "KIT esmfold2 n=$n exit=$([ $S = ok ] && echo 0 || echo $S) fwd=[$FWD] warm_median=${WARM:-NA} templates=[none: no template input] msa=[$(grep -aoE 'depth\(rows\) [0-9]+' $L | head -1)] process_rc=$RC" | tee -a $D/logs/all.results
  echo "job=${SLURM_JOB_ID:-} host=$(hostname) single-process KIT esmfold2 n=$n status=$S fwd=[$FWD] warm_median=${WARM:-NA}" > $DONE/kit_esmfold2_x$n
done
