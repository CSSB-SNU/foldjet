#!/bin/bash
# Copy of ~/foldforge-scratch/anthropic-kits/cu130kit.sh (the kits' run environment: Ubuntu 22.04 container, CUDA 13.0 forward-compat
# libcuda on the 575 driver, env passed by name) with singularity's cache/tmp moved under this directory (foldforge-scratch stays read-only).
S=/home/psk6950/foldforge-scratch
export SINGULARITY_CACHEDIR=/home/psk6950/miniworld-engine-scratch/bench5i28_1118_steady/openfold3_p2/tmp/sing_cache SINGULARITY_TMPDIR=/home/psk6950/miniworld-engine-scratch/bench5i28_1118_steady/openfold3_p2/tmp/sing_tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
C=$S/cudacompat13/usr/local/cuda-13.0/compat
exec singularity exec --nv --cleanenv --bind /home/psk6950 --bind /tmp --bind /var/spool/slurmd \
  --env LD_LIBRARY_PATH=$C:${CU130_EXTRA_LD:-}:/home/psk6950/miniworld-engine-scratch/bench5i28_1118_steady/openfold3_p2/syslib --env HOME=$HOME --env USER=$USER \
  $(for v in $(env | grep -E '^(XDG_|UV_|PIP_|TRITON_|TORCH|MODEL_OPT|HF_|JAX_|XLA_|CUDA_VISIBLE|SLURM_JOB_ID|PATH=|OMP_|MKL_|CUDA_|PYTHONNOUSERSITE|ODDE|OF3|RF3|BOLTZ|CHAI|PROTENIX|NVIDIA|FOLDFORGE|FF_|CUTLASS|MINIWORLD|FOUNDRY|ROSETTAFOLD3|OPENFOLD3|OPENFOLD_CACHE|PYTHONHASHSEED|PYTHONDONTWRITEBYTECODE|TMPDIR|CC=|CXX=)' | cut -d= -f1); do echo --env $v="${!v}"; done) \
  $S/sing/ubuntu2204.sif "$@"
