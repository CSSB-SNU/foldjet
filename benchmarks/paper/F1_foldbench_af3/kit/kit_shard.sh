#!/bin/bash
# F1, Anthropic kit side: the af3_torch kit as released, one process over one shard of FoldBench on one GPU (run inside cu130run.sh by run_kit.sbatch).
#   kit_shard.sh SHARD            env: KIT_ROOT KIT_INPUTS OUT [SHARD_LIST] [BIG=1]
# Kit-native modes only: --mode fast over the shard; if BIG=1, every target without a model.cif after the fast pass is run again in --mode big in a
# fresh kit process (the kit's route for OOM). No other retry: a target that fails stays failed (it is a result).
# Changes to the kit: none in its run path, except (1) the AF3 weight layout of the copy (xfold.of3.OF3 = False, two lines) with DeepMind's weights,
# (2) opt/af3_torch_opt/featurise.py passes ref_max_modified_date (upstream run_alphafold.py does; original kept as featurise.py.orig_kit).
# The kit takes FoldJet's AF3 JSON (MSAs + templates inside, hence --norun_data_pipeline): build them with build_inputs.py.
# Kit defaults kept: 10 recycles (11 trunk passes), 200 steps, 5 samples per seed; the seeds are the JSON's modelSeeds.
SH=$1
KIT_ROOT=${KIT_ROOT:?KIT_ROOT=<dir of the kit installs, as in benchmarks/5i28_kits>}
KIT_INPUTS=${KIT_INPUTS:?KIT_INPUTS=<dir with <target>.json of every target, from build_inputs.py>}
OUT=${OUT:?OUT=<run dir>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHARD_LIST=${SHARD_LIST:-$HERE/../shards/s$SH.txt}
W=$KIT_ROOT/af3_torch; K=$W/af3_torch_af3w
export XDG_CACHE_HOME=$W/cache UV_CACHE_DIR=$W/cache/uv PIP_CACHE_DIR=$W/cache/pip HF_HOME=$W/cache/hf TMPDIR=$W/tmp
export MODEL_OPT_JIT_ROOT=$W/jit TRITON_CACHE_DIR=$W/jit/triton_x TORCH_EXTENSIONS_DIR=$W/jit/torch_ext TORCHINDUCTOR_CACHE_DIR=$W/jit/inductor_x
mkdir -p "$MODEL_OPT_JIT_ROOT" "$TMPDIR"
export PYTHONHASHSEED=0 PYTHONFAULTHANDLER=1 CFLAGS=-g0
export CUDA_HOME=$KIT_ROOT/cuda13home T=${CUDA13TK:?CUDA13TK=<pixi env of the cu130 toolkit, scratch/cuda13tk/.pixi/envs/default>}
export CC=$T/bin/gcc CXX=$T/bin/g++
export PATH=$W/kitvenv/bin:$CUDA_HOME/bin:/usr/bin:/bin:/usr/local/bin
export AF3_TORCH_PY=$W/torch_venv/bin/python AF3_TORCH_JAX_PY=$W/alphafold3_venv/bin/python AF3_TORCH_JAX_REPO=$W/alphafold AF3_TORCH_PARAMS_DIR=$W/weights_af3
export PYTHONPATH=$K/opt:$K/opt/forward/af3t/af3_torch${PYTHONPATH:+:$PYTHONPATH}
. "$W/kitvenv/bin/activate"
D=$OUT/s$SH; I=$D/inputs
mkdir -p "$D"; rm -rf "$I"; mkdir -p "$I"
for t in $(cat "$SHARD_LIST"); do ln -s "$KIT_INPUTS/$t.json" "$I/$t.json"; done
cd "$K" || exit 1
{ echo "shard=$SH targets=$(ls "$I" | wc -l) host=$(hostname) job=${SLURM_JOB_ID:-}"; nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader
  python -c "import af3_torch_opt;print('af3_torch_opt from',af3_torch_opt.__file__)"; } > "$D/run_info.txt"
# the kit removes its work dir (with the per-seed forward.log: forward_s, tokens, failures) when it ends cleanly: keep a copy while it runs
keep() { while sleep 30; do cp -f "$1/.af3_torch_opt_work/forward.log" "$2" 2>/dev/null; done; }
S=$(date +%s.%N); echo "start $S" >> "$D/run_info.txt"
keep "$D/out" "$D/forward_fast.log" & KP=$!
bash run.sh pred --config h100 --mode fast --input_dir "$I" --output_dir "$D/out" --norun_data_pipeline > "$D/pred.log" 2>&1
rc=$?; E=$(date +%s.%N)
kill $KP 2>/dev/null; cp -f "$D/out/.af3_torch_opt_work/forward.log" "$D/forward_fast.log" 2>/dev/null
echo "end $E rc $rc" >> "$D/run_info.txt"
IB=$D/inputs_big; rm -rf "$IB"; mkdir -p "$IB"
for t in $(ls "$I"); do t=${t%.json}; [ -f "$D/out/$t/${t}_model.cif" ] || ln -s "$(readlink -f "$I/$t.json")" "$IB/$t.json"; done
NB=$(ls "$IB" | wc -l); echo "fast_end $(date +%s.%N) targets_without_model $NB: $(ls "$IB" | sed 's/.json//' | tr '\n' ' ')" >> "$D/run_info.txt"
if [ "${BIG:-1}" = 1 ] && [ "$NB" -gt 0 ]; then
  keep "$D/out_big" "$D/forward_big.log" & KP=$!
  bash run.sh pred --config h100 --mode big --input_dir "$IB" --output_dir "$D/out_big" --norun_data_pipeline > "$D/pred_big.log" 2>&1
  rc=$?
  kill $KP 2>/dev/null; cp -f "$D/out_big/.af3_torch_opt_work/forward.log" "$D/forward_big.log" 2>/dev/null
  echo "big_end $(date +%s.%N) rc $rc" >> "$D/run_info.txt"
fi
