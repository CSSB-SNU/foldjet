# Source this: environment of the paper training experiments (F6, F7, F10) on one card.
#   CARD=h100|a100|rtx5090   (default: guessed from the host name: cssb-rtx -> rtx5090, cssb-master3 / gpu* -> a100, else h100)
#   FF_REPO      this FoldJet checkout (default: the one holding this file)
#   FFENV        pixi env of a FoldJet checkout (its python is called directly: NEVER `pixi run` on a shared env)
#   ENGINE_SRC   optional foldjet-engine SOURCE tree put in front of the installed engine (a release candidate, e.g. rc-3.0.6)
#   PAPER_CACHE  inductor / XDG cache of the runs (WARM: reuse it, never a fresh dir per run; TRITON_CACHE_DIR is not set on purpose)
#   PAPER_OUT    output root (default ~/foldforge-scratch/paper on the H100 cluster; <id>/<card>/ below it)
# Nothing here starts python. Never run the scripts on a login node: sbatch / srun (H100, A100) or ~/mwe/gpu.sh (5090).
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FF_REPO=${FF_REPO:-$(cd "$_here/../../.." && pwd)}
if [ -z "${CARD:-}" ]; then
  case $(hostname) in
    cssb-rtx*) CARD=rtx5090 ;;
    DCTN-*) CARD=b200 ;;
    cssb-master3*|gpu0*|gpu1*) CARD=a100 ;;
    *) CARD=h100 ;;
  esac
fi
export CARD
export PAPER_OUT=${PAPER_OUT:-$HOME/foldforge-scratch/paper}
case $CARD in
  h100)
    export FFENV=${FFENV:-$FF_REPO/.pixi/envs/default}   # the checkout's own env (paper-306: engine 3.0.6)
    export PAPER_CACHE=${PAPER_CACHE:-$HOME/foldforge-scratch/train_speed/cache}
    export XDG_CACHE_HOME=$PAPER_CACHE TORCHINDUCTOR_CACHE_DIR=$PAPER_CACHE/inductor
    ;;
  a100)
    export FFENV=${FFENV:-$HOME/practice/ff_train_a9e1549/.pixi/envs/default}
    export PAPER_CACHE=${PAPER_CACHE:-$HOME/practice/paper_train_cache}
    export XDG_CACHE_HOME=$PAPER_CACHE TORCHINDUCTOR_CACHE_DIR=$PAPER_CACHE/inductor MINIWORLD_ENGINE_TARGET_ARCH=sm80 MAX_JOBS=4
    ;;
  rtx5090)
    export FFENV=${FFENV:-$HOME/ff/FoldJet/.pixi/envs/default}
    export FOLDFORGE_HOME=${FOLDFORGE_HOME:-$HOME/ff/home}
    export PAPER_CACHE=${PAPER_CACHE:-$HOME/mwe/jit/rc306ff}
    export XDG_CACHE_HOME=$PAPER_CACHE/xdg TORCHINDUCTOR_CACHE_DIR=$PAPER_CACHE/inductor
    export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH MINIWORLD_ENGINE_HEADER_CACHE=$HOME/.cache/miniworld_engine_headers
    export MINIWORLD_ENGINE_BUILD_ARCH=sm120 MAX_JOBS=4 TORCHINDUCTOR_COMPILE_THREADS=4 PAPER_OUT=${PAPER_OUT_5090:-$HOME/ff/paper}
    ;;
  b200)
    # host `b200` (no Slurm, GPUs 0-3 only; the lane driver sets CUDA_VISIBLE_DEVICES): everything from common/cards/b200.env (engine 3.0.7 source ENGINE_ROOT,
    # sm100a kernels from the prebuilt JIT cache under XDG_CACHE_HOME = $B200_P/cache, which must NOT be redirected to a per-card cache)
    unset PAPER_OUT   # env.sh set the cluster default above; b200.env sets $B200_P/ff/paper
    source "$_here/../common/cards/b200.env"
    export FFENV=${FFENV:-$FF_ENV} ENGINE_SRC=${ENGINE_SRC:-$ENGINE_ROOT}
    export PAPER_CACHE=${PAPER_CACHE:-$B200_P/cache/f6}
    ;;
  *) echo "unknown CARD=$CARD" >&2; return 1 ;;
esac
export PATH=$FFENV/bin:$PATH LD_LIBRARY_PATH=$FFENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ENGINE_SRC:+$ENGINE_SRC/src:}$FF_REPO/src:$FF_REPO/benchmarks/paper/train_common${PYTHONPATH:+:$PYTHONPATH}
export FF_ALLOW_SIZE_MISMATCH=1 PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PAPER_CACHE" 2>/dev/null
