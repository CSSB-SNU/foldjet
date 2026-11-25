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
  FoldJet with `language_model.allow_no_protein`, which is **not in `fdad9e2`** (it exists only as an uncommitted change in the main checkout, 2026-11-21): those
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

Run directory `~/foldforge-scratch/paper/F4/<card>/{rows,pred,specs,meta}/`; each row has FoldJet commit, engine version / source, GPU, host, Slurm job.
`data/F4_times_<card>.csv`: `model,target,task,tokens,anthropic,engine,torch,engine_fp8` (seconds; = the shape of `tables/supp_foldbench_times.csv`) + `torch_over_engine`,
`torch_over_engine_fp8`, peak memory and run information per arm. `data/F4_failures_<card>.csv`: every failed cell with status, OOM flag and reason.

## A100 data (engine 3.0.6, cssb3, 2026-11-22)

`data/F4_times_a100.csv` (297 rows = 11 predictors x 27 targets, arms torch / engine; FP8 is H100 only) and `data/F4_failures_a100.csv`. Column
`engine_cuda_only`: `yes` when the run's `BACKEND_TABLE` has `all_cuda: true`. On engine 3.0.6 OpenDDE was `no` (every `TokenDiTProtenix` call of the
refiner carries `extra_pair_bias`, which the sm_80 kernels of 3.0.6 refused: 9,600 calls on Triton, job 97152); its engine rows are now engine 3.0.8 (next
section). Failures: none for the engine since the ESMFold2 rerun of the 6 nucleic-acid-only targets (allow_no_protein, 22febf7); torch = 52 out-of-memory cells.

## OpenDDE engine arm re-measured on engine 3.0.8: A100 and RTX 5090 (2026-11-22)

Engine 3.0.8 serves `TokenDiTProtenix` / `TokenDiTMiniWorld` with `extra_pair_bias` on cuda on sm_80 and sm_120 (3.0.6 refused it -> Triton). Only the
OpenDDE engine arm of these two cards was re-run; every other predictor and arm keeps its engine-3.0.6 rows. Same protocol, node and FoldJet checkout as
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

Peak memory (`engine_peak_gib`) of the OpenDDE engine rows on 3.0.8 includes a resident +0.44 GiB at every size on both cards (A100 3.88 -> 4.33 GiB at 28
tokens, 63.30 -> 63.74 at 2,008): the cuda `extra_pair_bias` path keeps per-module weight packs in its resident cache (18.6 MiB per TokenDiT block x 24 =
0.436 GiB, independent of L; on 3.0.6 Triton served these calls without it; engine session's diagnosis). On the 32 GB RTX 5090 this pushes **7wvs (1,064
tokens) out of memory on 3.0.8** (3.0.6: 30.08 GiB peak, 119.9 s; short by ~0.1 GiB); reproduced in a fresh process (separate output dir), and both runs
already used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (as the whole campaign), so the cell stays OOM: cause = the resident weight packs. The 5090
OpenDDE engine arm therefore has 24 / 27 cells on 3.0.8 (7wvs, 8xgu, 8j62 OOM; 3.0.6: 25 / 27). F5 (sample comparison) still uses the
engine-3.0.6 OpenDDE predictions of these cards (not recomputed: not in the requested scope).

## H100: ESMFold2 re-measured with the release sampler (134 steps), engine 3.0.9 (2026-11-23)

FoldJet ran ESMFold2 with all 200 diffusion steps; the release (and the kit) caps the noise schedule at max_inference_sigma = 256 A, i.e. 134
steps (fix 5553d19). Every ESMFold2 cell of the H100 table was re-run on FoldJet main 49d41bb, engine 3.0.9, node02 (Slurm array 65821, one
(arm, target) per task): 108 / 108 ok, including the 6 nucleic-acid-only targets (allow_no_protein). `data/F4_times_h100_esmfold2_134.csv` replaces
the ESMFold2 rows of `data/existing/F4_h100_engine3.0.1.csv` (200 steps, engine 3.0.1). Engine arms `all_cuda: true` on all 27 targets; the Anthropic
arm keeps its kit refusals (fp32 Transition -> engine PyTorch backend; recorded, not gated).

| tokens | engine (s) 200 st. / 3.0.1 -> 134 st. / 3.0.9 | FP8 | PyTorch | Anthropic |
|---|---|---|---|---|
| 107 (9bks) | 2.08 -> 1.50 | 2.13 -> 1.53 | 6.00 -> 3.98 | 6.05 -> 4.24 |
| 658 (8ts0) | 3.17 -> 2.61 | 3.21 -> 2.58 | 26.65 -> 21.69 | 15.19 -> 10.79 |
| 2008 (8j62) | 13.54 -> 12.74 | 11.32 -> 10.44 | 166.81 -> 140.35 | 76.46 -> 56.11 |

F5 pairs of the new runs: `../F5_sample_comparison/data/F5_h100_esmfold2_134_pairs_summary.csv` (median RMSD to PyTorch sample k: engine 0.74 A,
FP8 0.96 A, Anthropic 0.65 A; PyTorch sample-to-sample baseline 2.65 A). The 200-step files of 2026-11-22 (`F4_times_h100_esmfold2_np.csv`,
`F5_h100_esmfold2_np_pairs_summary.csv`) were removed. F9 (inference limit) is not re-run: the number of sampler steps does not change the peak memory.
A100: the ESMFold2 cells are re-measured by another session (cssb3); RTX 5090: ESMFold2 not feasible (16 GB RAM).

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldJet `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F4 (this folder):** `data/F4_times_b200.csv` (297 rows = 11 predictors x 27 targets, arms `torch` / `engine`; no FP8 arm), `data/F4_failures_b200.csv` (35 cells), `data/F4_backend_engine_b200.csv` (per engine run: served / refused calls, `engine_cuda_only`). Failures: 23 PyTorch-operator out-of-memory cells (valid results, `oom = True`: OpenDDE 9, IntelliFold2 4, Chai-1 2, Protenix v2 2, one each AF3, Boltz2, OpenFold3, OpenFold3-preview2, Protenix v1, RoseTTAFold3) and ESMFold2 on the 6 nucleic-acid-only targets (`No protein token reached the language model`, both arms: 12 cells, the same as on the A100; not out-of-memory). Every other engine cell is finite. Engine label: OpenDDE = A, the other predictors = B (`engine_label`).
**OpenDDE TokenDiTProtenix `extra_pair_bias`:** the released 3.0.7 / clamp-only tree refused it on the B200 as on the A100 (9,600 Triton calls: 'inference: outside the fused H100 inference block's range ...'); with the B200 holes fix (A) the OpenDDE engine arm has Triton 8 calls = the LayerNorm w64 exception only. The clamp-only OpenDDE rows are not in the CSV (`invalid_archive`).
ESMFold2 loads ESM-C on the B200 without the 16 GB host-RAM limit of the 5090 and runs (ESM-C 6B fp32); on the 5090 ESMFold2 / ESMFold2-Fast are not feasible (the 24 GB fp32 safetensors exceed the 16 GB RAM cap; killed by systemd-oomd).
**Engine version (2026-11-22):** FoldJet is pinned to engine 3.0.9 (3.0.9 changes only B200 code paths relative to 3.0.8; the A100 / RTX 5090 / H100 campaign rows keep their measured versions 3.0.1 / 3.0.2 / 3.0.6 / 3.0.8 and are not re-measured); the B200 rows were measured on source trees containing the 3.0.9 code (labels in the CSVs: `engine_label`, `notes`).

## B200: ESMFold2 re-measured with the release sampler (134 steps), engine a553001e1 (2026-11-23)

`data/F4_times_b200_esmfold2_134.csv` replaces the ESMFold2 rows of `data/F4_times_b200.csv` (200 steps, engine label B). FoldJet ed6efa8b3 +
22febf7 (allow_no_protein) + 5553d19 (max_inference_sigma 256) as a working-tree patch (verified: 134 steps, first sigma 256), engine main
a553001e1 for the engine arm, arms engine / torch, 27 / 27 targets ok on both arms (the 6 nucleic-acid-only targets, failures in the campaign,
now run; `data/F4_failures_b200_esmfold2_134.csv` is empty). Engine 1.20-7.31 s (8j62 2,008 tokens: 7.31 s, was 8.13 at 200 steps),
PyTorch 2.87-95.3 s. Engine arm backend: all cuda except the declared B200 LayerNorm exception (width 256 bf16, >= 384^2 rows → Triton).

## RTX 5090: engine 3.0.10 arm (measured on branch 5090-int) + FP8 arm + Anthropic arm (DATA_STATUS, 2026-11-24)

**Released as 3.0.10 (2026-11-24):** engine `5090-int` 07bf36ae4 -> v3.0.10 changes no code the RTX 5090 runs (TokenDiTMiniWorldBiasOnly `backend_versions` number, the GPU-less prebuild plan, an A100-only `MWE_SM120` guard in `wide_ops.cu`), so these rows are the 3.0.10 rows; the CSV `engine` columns say so and keep the measured commit.

**Engine:** branch `5090-int` 07bf36ae4 as source (`ENGINE_SRC`): 3.0.9 + origin/main + Boltz-2 up-gate transition and token pair-bias hoist on sm_120
+ the RTX 5090 FP8 routes (off unless `MINIWORLD_FP8` is set). **Unreleased.** FoldJet 49d41bb (differs from the 3.0.6 campaign's 6a0593e only in
ESMFold2 code). Protocol of `rtx5090.sh` unchanged (27 targets, seeds 0, 200 steps, 4 trunk passes, 5 samples, bf16, eager, one process per
(predictor, arm), warm median of cold + warm). ESMFold2 / ESMFold2-Fast: not feasible on the 5090 (16 GB host RAM), as before. OpenFold3-preview2:
not in the 5090 campaign (as before).

**(1) engine bf16, cssb-rtx (the box of the old rows), real weights** — `data/F4_times_rtx5090_3010.csv` (243 rows; `s` new, `s_prev` the
released row: 3.0.6, OpenDDE 3.0.8; `prev_over_s` = old / new, > 1 = faster; peak memory both; PyTorch time of the 3.0.6 campaign for reference).
`data/F4_times_rtx5090.csv` is unchanged. Runs 2026-11-24 09:26-11:07 KST under the GPU lock; every run `all_cuda: true`; the out-of-memory cells are
exactly the old ones (no new OOM, none recovered).

| predictor | sum old -> new (s) | ratio | median per target | >= 539 tokens | peak (median, max) |
|---|---|---|---|---|---|
| AF3 | 202.2 -> 191.7 | 1.06 | 0.98 | 1.05-1.10 | +4 %, +17 % |
| Boltz-2 | 257.2 -> 224.5 | 1.15 | 1.14 | 1.11-1.32 | +15 %, +17 % |
| Chai-1 | 172.5 -> 166.4 | 1.04 | 0.99 | | +1 %, +10 % |
| IntelliFold2 | 269.5 -> 256.7 | 1.05 | 1.00 | | +1 %, +4 % |
| OpenDDE (old 3.0.8) | 279.9 -> 270.1 | 1.04 | 0.99 | | +4 %, +18 % |
| OpenFold3 | 202.5 -> 192.9 | 1.05 | 0.97 | | +4 %, +17 % |
| Protenix v1 | 194.9 -> 184.3 | 1.06 | 0.98 | | +4 %, +15 % |
| Protenix v2 | 231.2 -> 217.1 | 1.07 | 0.99 | | +3 %, +9 % |
| RoseTTAFold3 | 208.3 -> 198.5 | 1.05 | 0.97 | | +4 %, +13 % |

**Small targets (<= 450 tokens, eager, host-bound): 0.96-1.00x of the 3.0.6 rows.** The campaign overlapped with CPU test jobs of another agent on
cssb-rtx, so the 17 small targets were re-run for all 9 predictors on a CPU-quiet cssb-rtx (12:10 KST), interleaved per predictor with the released
3.0.9 source as a same-session control, and then AF3 / OpenFold3 again against the v3.0.6 source (git tag) in the order 3.0.6 / new / 3.0.6:
- re-run = campaign (median campaign / re-run 0.99-1.02): the campaign cells were not contaminated and are kept;
- new / released 3.0.9, same session: 1.00 (0.99-1.01) for every predictor except Boltz-2 (1.10 faster) and OpenDDE (0.987);
- new vs v3.0.6, same session: AF3 0.972 (0.965-0.998), OpenFold3 0.988 (0.973-1.016) (3.0.6 / 3.0.6 repeat 0.99).
So the small-target loss of 1-3 % is between 3.0.6 and 3.0.9 (already in the released 3.0.9), not from this branch; it is open (engine side, not
profiled yet). Large targets are 5-32 % faster.

**(2) engine FP8 + same-machine engine bf16 control, rented 4x RTX 5090 (`vast5090`, 575 W, Xeon Gold 6530 host), real weights** — the FP8 arm
(`MINIWORLD_FP8=safe` + `MINIWORLD_FP8_MODEL=<predictor>`, lazy guards, `fp8.check()` after each forward with up to 3 redo = the H100 protocol;
on the 5090 `safe` = the `SM120_ROUTES` slots TriAttn, TriMul, OPM, PWA, token DiT, atom DiT; the Transition is excluded from the preset) and the bf16 control were run on the same rented node,
4 jobs at a time (one per GPU) from one queue. `data/F4_times_rtx5090_fp8_3010.csv`. That host is ~2.7x slower than cssb-rtx on the
host-bound small targets (AF3 8gzl 8.8 s vs 3.2 s) and equal on the GPU-bound ones (8j62 39.5 vs 40.0 s): **compare FP8 with the vast bf16 column, not
with the cssb-rtx rows.**

**(3) Anthropic arm: not runnable on sm_120.** The release tree (f4f62fa6) was put on the rented node and smoke-tested (one target per predictor,
2026-11-24 12:50 KST): AF3, Boltz-2, OpenFold3, IntelliFold2, RoseTTAFold3 `OutOfResources: shared memory, Required 196,608-198,656, Hardware limit
101,376` (the release's Triton kernels are sized for the H100's 228 KB); Protenix v2 `Required 108,544`; Protenix v1, OpenDDE, Chai-1 `BackendUnsupported:
Transition backend 'anthropic': the provider served row 'pf' / 'lnl', its selection was 'v2'`. Recorded as the result (external backend: connected,
not modified).

FP8 vs bf16 on the rented node (`bf16 / FP8` > 1 = FP8 faster; sums over the targets both arms completed; >= 992 tokens = the GPU-bound targets):

| predictor | targets | sum bf16 / FP8 (s) | median per target | >= 992 tokens | FP8 peak vs bf16 (median) | not ok bf16 / FP8 | FP8 redo |
|---|---|---|---|---|---|---|---|
| AF3 | 27 | 322.5 / 316.9 = 1.02 | 0.98 | 1.00-1.17 | -3.0 % | 0 / 0 | 0 |
| Boltz-2 | 27 | 328.7 / 333.1 = 0.99 | 0.94 | 1.01-1.10 | -0.4 % | 0 / 0 | 0 |
| Chai-1 | 25 | 256.4 / 239.4 = 1.07 | 1.04 | 1.08-1.14 | -7.5 % | 2 / 1 (8xgu runs on FP8) | 0 |
| IntelliFold2 | 23 | 373.2 / 365.4 = 1.02 | 1.00 | 1.06-1.07 | -1.7 % | 4 / 4 | 0 |
| OpenDDE | 24 | 374.9 / 371.8 = 1.01 | 0.97 | 1.05-1.15 | -1.9 % | 3 / 3 | 0 |
| OpenFold3 | 27 | 319.8 / 306.5 = 1.04 | 1.01 | 1.06-1.18 | -3.0 % | 0 / 0 | 0 |
| Protenix v1 | 27 | 310.2 / 295.4 = 1.05 | 1.01 | 1.06-1.19 | -3.2 % | 0 / 0 | 0 |
| Protenix v2 | 26 | 334.1 / 325.2 = 1.03 | 1.00 | 1.03-1.10 | -2.8 % | 1 / 1 | 0 |
| RoseTTAFold3 | 27 | 346.4 / 324.1 = 1.07 | 1.05 | 1.02-1.10 | -2.5 % | 0 / 0 | 0 |

Every not-ok cell is out of memory. No FP8 forward saturated (`fp8_redo` 0 everywhere). On this host the small targets are host-bound (eager) and FP8
gains nothing there (0.94-1.04); on the >= 992-token targets FP8 is 1.00-1.19x. Accuracy of the FP8 predictions: F5 README.
Machines: engine bf16 F4 (part 1) and the small-target re-runs = cssb-rtx; FP8 + bf16 control (part 2) and the Anthropic smoke (part 3) = vast5090.
