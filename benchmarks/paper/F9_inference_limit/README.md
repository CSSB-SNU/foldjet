# F9: inference limit, the largest 5I28 assembly one forward completes

Paper: new item. Cards: H100 (80 GB), A100 (80 GB), RTX 5090 (32 GB). Per predictor: engine bf16, engine FP8 (H100), PyTorch operators, Anthropic kit (fast / big).

## What is measured

The largest n (copies of 5I28 x 128 tokens) at which ONE forward completes with finite output, the peak memory at that size, and the first size that does not (and why).
Input = the F3 protocol (real MSA + 4 distinct real templates, 4 trunk passes, 200 steps, 5 samples, bf16). Based on the rc-3.0.6 AF3 limit sweeps (`lim_check.py` /
`job_h100.sh`), generalised to every predictor and arm.

- `run_size.py`: one probe = one fresh process, one forward (`benchmark_repeats` 0). `status`: `ok` (finite coordinates and pLDDT), `oom` (with the allocator numbers),
  `nonfinite`, `bucket` (a size beyond FoldForge's atom bucket ladder: 131,072 atoms since `9fd04e2`), `failed` (reason in `error`). Execution: engine arms = the 5I28
  benchmark's compile + CUDA graphs + bucketing; the torch arm = eager (PyTorch's operators as such, as F4); `--exec` overrides. Peak memory: the forward's own
  (`model_peak_*`) and the process peak.
- `limit_search.py`: sizes 16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 128 in order until the first that does not complete, then **bisection** by one copy between the last
  ok and that size, then 2 more step sizes past the edge to catch a non-monotonic edge (the RTX 5090 showed one with AF3 before the pair-band fix: n = 19-22 OOM, 23-24 ok).
  An OOM of the 'reserved but unallocated' kind is retried once with expandable segments (the 5I28 harness's own fallback). A failure that is not memory ends the search
  and is reported with its reason. Resumable (`sizes.jsonl`).
- Kit arm (`--kit K`): the kit's own `run.sh` through the harness `fb.sh` (fast, big on a fast OOM). `kit_inputs.py` writes the n-copy kit inputs beyond n = 16 into the
  harness (new files only; AF3 kits 1 seed; Protenix family keeps the harness' 7 seeds; ESMFold2 `FWD_CALLS=1`): automated for `af3_jax af3_torch openfold3 chai1 esmfold2
  esmfold2_fast boltz2 protenix_v1 protenix_v2 opendde`; **not** for `rf3` and `intellifold2` (threaded template PDB / overlay inputs per n): those two report `no_input`.
  RTX 5090 has no kit install.
- FP8 arm: `MINIWORLD_FP8=safe` + `MINIWORLD_FP8_MODEL`, lazy guards, saturation check recorded per probe (`fp8_saturated`).

## Commands

```bash
# smoke (a size list of two small sizes; the search logic is covered by the fake-probe test, see the commit notes)
MODEL=af3 ARM=engine STEPS="1 2" ENGINE_SRC=~/miniworld-engine-scratch/rc306/src sbatch --time=00:10:00 -J f9s --export=ALL benchmarks/paper/F9_inference_limit/h100.sbatch
# full campaign (not started by this branch)
CARD=h100 NODE=node02 benchmarks/paper/F9_inference_limit/submit_all.sh       # 33 FoldForge jobs (11 x engine / engine_fp8 / torch) + 10 kit jobs
CARD=a100 benchmarks/paper/F9_inference_limit/submit_all.sh                   # on cssb3
for m in protenix1 protenix2 opendde esmfold2 esmfold2-fast intellifold2 af3 boltz2 chai1 openfold3 rosettafold3; do for a in engine torch; do
  MODEL=$m ARM=$a STEPS="8 16 20 24 32 40 48" benchmarks/paper/F9_inference_limit/rtx5090.sh 28800; done; done    # cssb-rtx, under the GPU lock
python benchmarks/paper/F9_inference_limit/collect.py --card h100
```

Each job is ONE continuous search (never chunked). Expected wall time per (predictor, arm): ~11 probes x (model load + compile + a forward of 1-10 min at n >= 24) = 1-3 h on the
H100 (an OOM probe is quick); the kit arm 1-5 h; A100 about 1.7x; 5090 fewer probes (edge near n = 24) ~1 h.

## Output

`~/foldforge-scratch/paper/F9/<card>/<arm>/<predictor>/{sizes.jsonl,limit.json,cells/n<N>.json,run.log}`; `data/F9_<card>.csv` (one row per predictor x arm: largest ok n and tokens,
first not ok + status, non-monotonic flag, peak memory at the edge, stop reason, run information) and `data/F9_<card>_sizes.csv` (every probe).
