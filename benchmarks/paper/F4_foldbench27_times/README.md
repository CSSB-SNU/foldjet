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

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldForge `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F4 (this folder):** `data/F4_times_b200.csv` (297 rows = 11 predictors x 27 targets, arms `torch` / `engine`; no FP8 arm), `data/F4_failures_b200.csv` (35 cells), `data/F4_backend_engine_b200.csv` (per engine run: served / refused calls, `engine_cuda_only`). Failures: 23 PyTorch-operator out-of-memory cells (valid results, `oom = True`: OpenDDE 9, IntelliFold2 4, Chai-1 2, Protenix v2 2, one each AF3, Boltz2, OpenFold3, OpenFold3-preview2, Protenix v1, RoseTTAFold3) and ESMFold2 on the 6 nucleic-acid-only targets (`No protein token reached the language model`, both arms: 12 cells, the same as on the A100; not out-of-memory). Every other engine cell is finite. Engine label: OpenDDE = A, the other predictors = B (`engine_label`).
**OpenDDE TokenDiTProtenix `extra_pair_bias`:** the released 3.0.7 / clamp-only tree refused it on the B200 as on the A100 (9,600 Triton calls: 'inference: outside the fused H100 inference block's range ...'); with the B200 holes fix (A) the OpenDDE engine arm has Triton 8 calls = the LayerNorm w64 exception only. The clamp-only OpenDDE rows are not in the CSV (`invalid_archive`).
ESMFold2 loads ESM-C on the B200 without the 16 GB host-RAM limit of the 5090 and runs (ESM-C 6B fp32).
