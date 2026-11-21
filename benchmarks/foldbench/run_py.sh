#!/bin/bash
# The python of a FoldForge pixi env directly (never `pixi run`: it re-syncs a shared env to the lock), this checkout's src on PYTHONPATH.
#   benchmarks/foldbench/run_py.sh SCRIPT.py ARGS...
# FFENV       the env (default <repo>/.pixi/envs/default)
# ENGINE_OVERLAY  optional: dir of an extracted engine wheel put first on PYTHONPATH (release-candidate runs; default: the env's engine)
# CACHE_ROOT  kernel / compile caches (default ${FOLDFORGE_HOME:-~/.cache/foldforge}/cache/foldbench): XDG_CACHE_HOME, TORCH_EXTENSIONS_DIR, TORCHINDUCTOR_CACHE_DIR
# Weights come from FOLDFORGE_CHECKPOINT_DIR / FOLDFORGE_HOME as usual (docs/guides/CONFIG.md). PYTORCH_CUDA_ALLOC_CONF: FoldForge's default
# (expandable_segments:True) unless set (docs/guides/CONFIG.md).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
E=${FFENV:-$REPO/.pixi/envs/default}
C=${CACHE_ROOT:-${FOLDFORGE_HOME:-$HOME/.cache/foldforge}/cache/foldbench}
mkdir -p "$C"
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib:${LD_LIBRARY_PATH:-}
# ENGINE_OVERLAY: a directory with an extracted miniworld_engine wheel (+ kernels wheels) in front of the env's installed engine (release-candidate runs)
export PYTHONPATH=${ENGINE_OVERLAY:+$ENGINE_OVERLAY:}$REPO/src PYTHONFAULTHANDLER=1
export XDG_CACHE_HOME=$C/cache TORCH_EXTENSIONS_DIR=$C/cache/torch_extensions TORCHINDUCTOR_CACHE_DIR=$C/inductor MAX_JOBS=${MAX_JOBS:-8}
SCRIPT="$1"; shift
case "$SCRIPT" in /*) ;; *) [ -f "$SCRIPT" ] || SCRIPT=$HERE/$SCRIPT ;; esac
exec python "$SCRIPT" "$@"
