#!/bin/bash
# CPU-only check (no GPU): both kits' featurisers on the new inputs. Runs inside cu130run.sh:
#   srun -p h100 -A cssb --qos=normal_h100 -n 1 -c 8 --mem=48G --time=00:30:00 $KITS_DIR/cu130run.sh bash cpu_featprobe.sh 1 2 4 8 12 16
B=$BENCH_DIR; K=$KITS_DIR
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TMPDIR=$B/af3_jax/tmp/probe XDG_CACHE_HOME=$B/af3_jax/tmp/probe/xdg PYTHONDONTWRITEBYTECODE=1
for n in "$@"; do
  $K/af3_jax/venv/bin/python -I $B/af3_jax/featprobe.py $B/af3_jax/inputs/x$n/5i28.json 1 af3_jax 2>&1 | grep -E 'FEATPROBE|Error|Traceback' 
  $K/af3_torch/alphafold3_venv/bin/python -I $B/af3_torch/featprobe.py $B/af3_torch/inputs/x$n/5i28.json 0 af3_torch 2>&1 | grep -E 'FEATPROBE|Error|Traceback'
done
