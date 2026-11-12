# sourced INSIDE the cu130 container: the Chai-1 kit's venv130 (as anthropic-kits/env.sh chai1 with VENVD=venv130 + real_one.sh's PATH/kalign),
# but every cache/tmp under bench5i28/chai1 (seeded once from the kit's cache130 / jit_venv130 copies); kit tree, venv and weights are only read.
# PYTHONPATH = this bench's shim (offline RCSB copy from inputs/cifs, chai_lab INFO logging, the forward-span timer).
K=$KITS_DIR; W=$K/chai1; H=$BENCH_DIR/chai1
T=$FF_SCRATCH/cuda13tk/.pixi/envs/default
. $W/venv130/bin/activate
export XDG_CACHE_HOME=$H/cache UV_CACHE_DIR=$H/cache/uv PIP_CACHE_DIR=$H/cache/pip TRITON_CACHE_DIR=$H/cache/triton TORCH_EXTENSIONS_DIR=$H/cache/torch_ext \
       TORCHINDUCTOR_CACHE_DIR=$H/cache/inductor MODEL_OPT_JIT_ROOT=$H/jit HF_HOME=$H/cache/hf MPLCONFIGDIR=$H/cache/matplotlib
export LD_LIBRARY_PATH=$W/venv130/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CFLAGS=-g0 TMPDIR=$H/tmp CHAI_DOWNLOADS_DIR=$W/weights
export PATH=$PATH:$T/bin:$FF_SCRATCH/kalign_build/install/bin CC=$T/bin/gcc CXX=$T/bin/g++ CUDA_HOME=$T
export REAL_CIF_DIR=$H/inputs/cifs PYTHONPATH=$H/shim${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $TMPDIR
