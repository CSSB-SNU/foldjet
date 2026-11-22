# F4: 11 predictors x 27 FoldBench targets, model time: engine kernels vs PyTorch operators

Paper: Supp 10, text. Cards: H100, A100, RTX 5090 (FP8 arm: H100). The same runs feed F5.

## What is measured

Warm model time (`model_seconds_warm_median` of one cold + one warm forward; CUDA-synchronized; no featurization / loading / writing) of
`run_foldbench27.py MODEL ARM` over the 27 targets of `targets27.tsv` (3 per FoldBench task: small / medium / large, 28 - 2,008 tokens), ascending size, one
process per (predictor, arm). Predictors (11): `af3 boltz2 chai1 esmfold2 intellifold2 opendde openfold3 openfold3-preview2 protenix1 protenix2 rosettafold3`.

- **Arms:** `torch` (backend `pytorch`), `engine` (backend `miniworld`, bf16 kernels), `engine_fp8` (H100; `MINIWORLD_FP8=safe` + model; sync guards; `fp8.check()`
  after every forward, a saturated forward is run again, at most 3 times: `fp8_redo`), `anthropic` (optional: the engine's connection to Anthropic's kernels).
- **Same inputs and seeds:** FoldBench's own inputs (`$FB_ROOT/inputs/<target>/data.yaml`: real MSA, templates where the checkpoint takes them; ESMFold2 and
  RoseTTAFold3 without), `trunk_seed = diffusion_seed = 0`, 200 steps, 4 trunk passes, 5 samples, bf16, mode exact. Eager on every arm (no compile, no CUDA
  graph): the arms differ only in the backend. The structures of the last forward are saved (`pred/<arm>/<model>/<target>/`) for F5.
- A failure (also out of memory) is a row with `status: error` and the reason; never retried here, never a number. ESMFold2 on RNA / DNA-only targets needs a
  FoldForge with `language_model.allow_no_protein`, which is **not in `fdad9e2`** (it exists only as an uncommitted change in the main checkout, 2026-11-21): those
  targets are recorded as `error: No protein token` until that change is committed; `run_foldbench27.py` retries such rows on the next run.
- RTX 5090: needs the FoldBench inputs / MSA / template LMDBs under `$FB_ROOT` (`~/data/foldbench`; 27 inputs + `msa/` + `template/`, ~2.2 GB, copied there 2026-11-21)
  and the checkpoints under `$FOLDFORGE_HOME` (`~/ff/home`; `openfold3.miniworld.pt` was missing there on 2026-11-21: `openfold3` / `openfold3-preview2` need it).

## Commands

```bash
# smoke, H100 (comma list -> export it, do not put it into --export)
export ONLY=7sxp-assembly1,7fxw-assembly1 ENGINE_SRC=~/miniworld-engine-scratch/rc306/src
MODEL=af3 ARM=engine sbatch --time=00:10:00 -J f4s --export=ALL benchmarks/paper/F4_foldbench27_times/h100.sbatch
# A100: same with a100.sbatch on cssb3.   5090: MODEL=af3 ARM=engine ONLY=... benchmarks/paper/F4_foldbench27_times/rtx5090.sh 1500
# full campaign (after the release is pinned; not started by this branch)
CARD=h100 NODE=node02 benchmarks/paper/F4_foldbench27_times/submit_all.sh      # 33 jobs: 11 predictors x torch / engine / engine_fp8
CARD=a100 benchmarks/paper/F4_foldbench27_times/submit_all.sh                  # on cssb3: 22 jobs
for m in af3 boltz2 chai1 esmfold2 intellifold2 opendde openfold3 openfold3-preview2 protenix1 protenix2 rosettafold3; do for a in torch engine; do
  MODEL=$m ARM=$a benchmarks/paper/F4_foldbench27_times/rtx5090.sh 14400; done; done     # on cssb-rtx: 22 runs under the GPU lock
python benchmarks/paper/F4_foldbench27_times/collect.py --card h100            # -> data/F4_times_h100.csv, F4_failures_h100.csv
```

Expected wall time per (predictor, arm) job (earlier 2.5.8 / 3.0.1 runs, H100): 20 min - 3 h (the PyTorch arm of the large targets and RoseTTAFold3 /
IntelliFold2 are the slow ones; the 2,008-token protein-RNA target dominates). `--deadline-s` (default off) can bound a run; a stopped run resumes. A100 about 1.7x, 5090 about 2x.

## Output

Run directory `~/foldforge-scratch/paper/F4/<card>/{rows,pred,specs,meta}/`; each row has FoldForge commit, engine version / source, GPU, host, Slurm job.
`data/F4_times_<card>.csv`: `model,target,task,tokens,anthropic,engine,torch,engine_fp8` (seconds; = the shape of `tables/supp_foldbench_times.csv`) + `torch_over_engine`,
`torch_over_engine_fp8`, peak memory and run information per arm. `data/F4_failures_<card>.csv`: every failed cell with status, OOM flag and reason.

## A100 data (engine 3.0.6, cssb3, 2026-11-22)

`data/F4_times_a100.csv` (297 rows = 11 predictors x 27 targets, arms torch / engine; FP8 is H100 only) and `data/F4_failures_a100.csv`. Column
`engine_cuda_only`: `yes` when the run's `BACKEND_TABLE` has `all_cuda: true`; **`no` for OpenDDE**: every `TokenDiTProtenix` call of the refiner carries
`extra_pair_bias`, which the sm_80 attention kernels do not take, so all 9,600 calls ran the Triton backend (job 97152; details in
`~/foldforge-scratch/paper/engine_issues/a100_opendde_tokendit_extra_pair_bias.md`). The OpenDDE engine times on the A100 are therefore not pure-CUDA
numbers (a CUDA hole to fix in the engine). Failures: engine = ESMFold2 on the 6 nucleic-acid-only targets (`No protein token reached the language model`);
torch = 52 out-of-memory cells + the same 6.
