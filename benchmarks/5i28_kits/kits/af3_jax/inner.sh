#!/bin/bash
# Runs INSIDE cu130run.sh (Ubuntu 22.04 + CUDA 13.0 forward-compat container), called by run.sh. usage: inner.sh N
# Same kit route as the earlier real-MSA rows (anthropic-kits/real/run_j.sh): run.sh pred --config h100 --variant p2 --mode fast,
# --num_recycles 3 (= 4 trunk passes, AF3 runs num_recycles+1), 5 diffusion samples, 200 steps (AF3 default), p2 weights.
# Only the input, the output dir, TMPDIR/XDG caches and the compile-cache root (a private copy of the kit's prebuilt jitroot) differ.
N=$1
D=$BENCH_DIR/af3_jax
W=$KITS_DIR/af3_jax                      # the kit install (read-only use)
IN=$D/inputs/x$N/5i28.json; OUT=$D/out/x$N
export XDG_CACHE_HOME=$D/cache UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip TMPDIR=$D/tmp/x$N
export JAX_COMPILATION_CACHE_DIR=$D/cache/jax TRITON_CACHE_DIR=$D/cache/triton HF_HOME=$D/cache/hf
export AF3_JAX_REPO=$W/alphafold AF3_JAX_PY=$W/venv/bin/python PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export AF3_JAX_PARAMS_ROOT=$W/weights AF3_JAX_CACHE_ROOT=$D/jitroot MODEL_OPT_JIT_ROOT=$D/jitroot
rm -rf $OUT $TMPDIR; mkdir -p $OUT $TMPDIR $D/cache
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
# MSA / template evidence with the kit's own featuriser (CPU, separate process, before and outside the timed model process)
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $W/venv/bin/python -I $D/featprobe.py $IN 1 af3_jax 2>&1 | grep -E 'FEATPROBE|Error|Traceback'
. $W/kitvenv/bin/activate
cd $W/af3_jax
bash run.sh pred --config h100 --variant p2 --mode fast --json_path $IN --output_dir $OUT \
   --num_recycles 3 --num_diffusion_samples 5 --model_dir=$AF3_JAX_PARAMS_ROOT/p2
RC=$?
echo $RC > $OUT/rc; echo "KITRC=$RC"
exit $RC
