#!/bin/bash
# Copy of $KITS_DIR/cu130run.sh (the kits' pinned stack: Ubuntu 22.04 SIF + CUDA 13.0 forward-compat libcuda),
# identical except that singularity's cache/tmp live under this kit's bench directory (foldforge-scratch stays read-only).
S=$FF_SCRATCH
H=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SINGULARITY_CACHEDIR=$H/sing/cache SINGULARITY_TMPDIR=$H/sing/tmp
C=$S/cudacompat13/usr/local/cuda-13.0/compat
exec singularity exec --nv --cleanenv --bind $HOME --bind /tmp --bind /var/spool/slurmd \
  --env LD_LIBRARY_PATH=$C:${CU130_EXTRA_LD:-} --env HOME=$HOME --env USER=$USER \
  $(for v in $(env | grep -E '^(XDG_|UV_|PIP_|TRITON_|TORCH|MODEL_OPT|HF_|JAX_|XLA_|CUDA_VISIBLE|SLURM_JOB_ID|PATH=|OMP_|MKL_|CUDA_|PYTHONNOUSERSITE|ODDE|OF3|RF3|BOLTZ|CHAI|PROTENIX|NVIDIA|FOLDFORGE|FF_|CUTLASS|MINIWORLD|PYTHONPATH|LD_PRELOAD_NONE)' | cut -d= -f1); do echo --env $v="${!v}"; done) \
  $S/sing/ubuntu2204.sif "$@"
