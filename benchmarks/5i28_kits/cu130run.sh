#!/bin/bash
# Run a command in Ubuntu 22.04 (glibc 2.35) with CUDA 13.0 forward-compat libcuda on the 575 driver.
# Matches the kits' pinned stack (cu130 wheels, glibc>=2.32 prebuilt .so). Use on a GPU node: cu130run.sh <cmd> [args]
S=$FF_SCRATCH
export SINGULARITY_CACHEDIR=$S/sing/cache SINGULARITY_TMPDIR=$S/sing/tmp
C=$S/cudacompat13/usr/local/cuda-13.0/compat
exec singularity exec --nv --cleanenv --bind $HOME --bind /tmp --bind /var/spool/slurmd \
  --env LD_LIBRARY_PATH=$C:${CU130_EXTRA_LD:-} --env HOME=$HOME --env USER=$USER \
  $(for v in $(env | grep -E '^(XDG_|UV_|PIP_|TRITON_|TORCH|MODEL_OPT|HF_|JAX_|XLA_|CUDA_VISIBLE|SLURM_JOB_ID|PATH=|OMP_|MKL_|CUDA_|PYTHONNOUSERSITE|ODDE|OF3|RF3|BOLTZ|CHAI|PROTENIX|NVIDIA|FOLDFORGE|FF_|CUTLASS|MINIWORLD|PYTHONPATH|LD_PRELOAD_NONE)' | cut -d= -f1); do echo --env $v="${!v}"; done) \
  $S/sing/ubuntu2204.sif "$@"
