# sourced INSIDE the cu130 container: the Boltz-2 kit's venv130 (as anthropic-kits/env.sh boltz2 with VENVD=venv130), but every cache/tmp
# under bench5i28/boltz2 (seeded once from the kit's cache130 / jit_venv130 copies); kit tree, venv and weights are only read.
K=$KITS_DIR; W=$K/boltz2; H=$BENCH_DIR/boltz2
T=$FF_SCRATCH/cuda13tk/.pixi/envs/default
. $W/venv130/bin/activate
export XDG_CACHE_HOME=$H/cache UV_CACHE_DIR=$H/cache/uv PIP_CACHE_DIR=$H/cache/pip TRITON_CACHE_DIR=$H/cache/triton TORCH_EXTENSIONS_DIR=$H/cache/torch_ext \
       TORCHINDUCTOR_CACHE_DIR=$H/cache/inductor MODEL_OPT_JIT_ROOT=$H/jit HF_HOME=$H/cache/hf
export LD_LIBRARY_PATH=$W/venv130/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 CFLAGS=-g0 TMPDIR=$H/tmp CU130=1 BOLTZ_CACHE=$W/weights
export PATH=$PATH:$T/bin CC=$T/bin/gcc CXX=$T/bin/g++ CUDA_HOME=$T
mkdir -p $TMPDIR
