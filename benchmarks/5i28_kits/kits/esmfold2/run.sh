#!/bin/bash
# usage: run.sh N  -- ESMFold2 kit (fast, variant full_msa), the 5I28 benchmark: real MSA (6789 rows + query) on every chain; ESMFold2 has no
# template input. 4 forwards (first cold), FWDTIME = synchronized builder.fold (the kit harness esm_fwd_kit_msa.py, read-only).
N=$1; K=$KITS_DIR; W=$K/esmfold2; D=$BENCH_DIR/esmfold2
cat > $D/tmp/inner_x$N.sh <<EOT
export XDG_CACHE_HOME=$D/cache UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip TMPDIR=$D/tmp
export TRITON_CACHE_DIR=$D/cache/triton TORCH_EXTENSIONS_DIR=$D/cache/torch_ext TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor
export MODEL_OPT_JIT_ROOT=$D/jitroot HF_HOME=$W/weights HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt PYTHONHASHSEED=0 NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export CC=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/gcc CXX=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/g++
. $W/venv130/bin/activate
cd $W/esmfold2
source configs/h100.env
export ESMFOLD2_OPT=fast ESMFOLD2_VARIANT=full_msa N_CHAINS=$N FWD_CALLS=4
python $W/esm_fwd_kit_msa.py
EOT
$K/cu130run.sh bash $D/tmp/inner_x$N.sh > $D/logs/x$N.log 2>&1
RC=$?
L=$D/logs/x$N.log
FWD=$(grep -oE '^FWDTIME [0-9]+ [0-9.]+' $L | awk '{print $3}' | tr '\n' ' ')
WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | sort -g | awk '{a[NR]=$1} END{if(NR) print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
echo "KIT esmfold2 n=$N exit=$RC fwd=[$FWD] warm_median=$WARM templates=[none: no template input] msa=[$(grep -oE 'depth\(rows\) [0-9]+' $L | head -1)] active=$(grep -c 'ACTIVE' $L)"
