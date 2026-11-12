#!/bin/bash
# Runs INSIDE cu130run.sh (Ubuntu 22.04 + CUDA 13.0 forward-compat container), called by run.sh. usage: inner.sh N
# Same kit route as the earlier real-MSA rows (anthropic-kits/real/inner_t.sh): run.sh pred --config h100 --mode fast
# --norun_data_pipeline --num_recycles 3 (= 4 trunk passes, num_recycles+1) --diffusion_steps 200 --num_diffusion_samples 5.
# Only the input, the output dir, TMPDIR/XDG caches and the JIT root (a private copy of the kit's prebuilt jit/) differ.
N=$1
D=$BENCH_DIR/af3_torch
R=$KITS_DIR; W=$R/af3_torch; KT=$W/af3_torch   # the kit install (read-only use)
IN=$D/inputs/x$N/5i28.json; OUT=$D/out/x$N
export XDG_CACHE_HOME=$D/cache UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip HF_HOME=$D/cache/hf TMPDIR=$D/tmp/x$N
export MODEL_OPT_JIT_ROOT=$D/jit TRITON_CACHE_DIR=$D/jit/triton_x TORCH_EXTENSIONS_DIR=$D/jit/torch_ext TORCHINDUCTOR_CACHE_DIR=$D/jit/inductor_x
export PYTHONHASHSEED=0 PYTHONFAULTHANDLER=1 CFLAGS=-g0 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME=$R/cuda13home T=$FF_SCRATCH/cuda13tk/.pixi/envs/default
export CC=$T/bin/gcc CXX=$T/bin/g++
export PATH=$W/kitvenv/bin:$CUDA_HOME/bin:/usr/bin:/bin:/usr/local/bin
export AF3_TORCH_PY=$W/torch_venv/bin/python AF3_TORCH_JAX_PY=$W/alphafold3_venv/bin/python AF3_TORCH_JAX_REPO=$W/alphafold AF3_TORCH_PARAMS_DIR=$W/weights
rm -rf $OUT $TMPDIR; mkdir -p $OUT $TMPDIR $D/cache
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
$AF3_TORCH_PY -I -c "import torch;print('torch',torch.__version__,torch.version.cuda)"
# MSA / template evidence with the kit's own featuriser env (CPU, separate process, before and outside the timed model process)
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= $AF3_TORCH_JAX_PY -I $D/featprobe.py $IN 0 af3_torch 2>&1 | grep -E 'FEATPROBE|Error|Traceback'
. $W/kitvenv/bin/activate
cd $KT
bash run.sh pred --config h100 --mode fast --json_path $IN --output_dir $OUT/out --norun_data_pipeline \
   --num_recycles 3 --diffusion_steps 200 --num_diffusion_samples 5
RC=$?
echo $RC > $OUT/rc; echo "KITRC=$RC"
exit $RC
