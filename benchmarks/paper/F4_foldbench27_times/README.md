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
`engine_cuda_only`: `yes` when the run's `BACKEND_TABLE` has `all_cuda: true`. On engine 3.0.6 OpenDDE was `no` (every `TokenDiTProtenix` call of the
refiner carries `extra_pair_bias`, which the sm_80 kernels of 3.0.6 refused: 9,600 calls on Triton, job 97152); its engine rows are now engine 3.0.8 (next
section). Failures: none for the engine since the ESMFold2 rerun of the 6 nucleic-acid-only targets (allow_no_protein, 22febf7); torch = 52 out-of-memory cells.

## OpenDDE engine arm re-measured on engine 3.0.8: A100 and RTX 5090 (2026-11-22)

Engine 3.0.8 serves `TokenDiTProtenix` / `TokenDiTMiniWorld` with `extra_pair_bias` on cuda on sm_80 and sm_120 (3.0.6 refused it -> Triton). Only the
OpenDDE engine arm of these two cards was re-run; every other predictor and arm keeps its engine-3.0.6 rows. Same protocol, node and FoldForge checkout as
the 3.0.6 run (A100: gpu05, Slurm 97288, checkout c129a81 = 6a0593e + allow_no_protein; 5090: cssb-rtx under the GPU lock, 6a0593e); the 3.0.8 package (the
H100 pixi install of the pin) was put in front of each card's environment (`ENGINE_SRC`), the warm caches kept. Rows: label engine 3.0.8 (`engine_*_version`
3.0.8), `engine_cuda_only = yes`, the 3.0.6 time in the new column `engine_3.0.6`.

Backend report, both cards: `all_cuda: true`; `TokenDiTProtenix` 24 modules, served cuda 9,600, refused none; Triton 0.

| tokens | A100 3.0.6 -> 3.0.8 (s) | ratio | RTX 5090 3.0.6 -> 3.0.8 (s) | ratio |
|---|---|---|---|---|
| 28-107 (6 targets) | 11.5-11.8 -> 8.6-8.7 | 1.32-1.36x | 3.8 -> 3.2-3.3 | 1.14-1.17x |
| 138-268 (7) | 12.1-16.6 -> 9.3-11.1 | 1.30-1.61x | 6.3-10.0 -> 4.0-5.6 | 1.56-2.10x |
| 358-658 (8) | 27.8-77.2 -> 11.2-25.3 | 2.26-3.29x | 19.3-53.4 -> 7.0-18.6 | 2.44-3.04x |
| 992-1188 (4) | 111.6-232.5 -> 40.1-70.9 | 2.78-3.28x | 82.9-168.4 -> 31.3-54.5 (7wvs: OOM) | 2.65-3.09x |
| 1487, 2008 (2) | 371.8 / 672.1 -> 114.2 / 210.4 | 3.26 / 3.19x | OOM on both versions | |

Peak memory is ~0.45 GiB higher on 3.0.8 at every size (A100 3.88 -> 4.33 GiB at 28 tokens, 63.30 -> 63.74 at 2,008). On the 32 GB RTX 5090 this pushes
**7wvs (1,064 tokens) out of memory** (3.0.6: 30.08 GiB peak, 119.9 s); reproduced in a fresh process (a separate output dir), so not fragmentation. The 5090
OpenDDE engine arm therefore has 24 / 27 cells on 3.0.8 (7wvs, 8xgu, 8j62 OOM; 3.0.6: 25 / 27). Reported to the engine session. F5 (sample comparison) still uses the
engine-3.0.6 OpenDDE predictions of these cards (not recomputed: not in the requested scope).

## H100: ESMFold2 on the 6 nucleic-acid-only targets (2026-11-22)

The H100 rows in `data/existing/` (engine 3.0.1) have ESMFold2 failing on 8gzl, 7sxp, 8fi1, 8d79, 7wii, 9g7c (`No protein token reached the language
model`). After the FoldForge fix (`allow_no_protein`, 22febf7: no protein token -> zero language-model hidden states) these 6 targets x 4 arms were re-run on
engine 3.0.7 (node01, Slurm 65132-65135 and 65144-65163, one target per job): 24 / 24 ok. `data/F4_times_h100_esmfold2_np.csv` (seconds, warm median):
engine 2.13-2.21, engine FP8 2.21-2.33, PyTorch 5.80-9.62, Anthropic kit 5.90-9.66. Engine arms: `all_cuda: true`. Anthropic arm: the kit refuses the
fp32 `Transition` cells (800 calls served by the engine PyTorch backend) and has no `AtomDiTESMFold2Pair` (PyTorch); recorded, not gated (baseline).
F5 pairs of these targets: `../F5_sample_comparison/data/F5_h100_esmfold2_np_pairs_summary.csv` (median RMSD to PyTorch: engine 0.99 A, FP8 1.03 A,
Anthropic 0.91 A; PyTorch sample-to-sample baseline 3.12 A).
