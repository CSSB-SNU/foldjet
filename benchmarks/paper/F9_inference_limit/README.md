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

## Status (engine 3.0.6 campaign, 2026-11-21)

- **H100** (`data/engine306/F9_h100.csv`, `F9_h100_sizes.csv`): engine bf16, engine FP8 and PyTorch operators for all 11 predictors, FoldForge `paper-306`
  (pinned engine 3.0.6). AF3 engine bf16 was confirmed only at its known edge (n = 36 ok / 37 OOM, `STEPS="36 37"`). Every completed engine cell ran all its
  engine classes on cuda (`all_cuda` true); every first failing size is an OOM. No non-monotonic edge on the H100.
- **Anthropic kit arm: not measured (user decision, 2026-11-21).** The kit jobs were cancelled; the partial kit probes of that campaign were moved out of the
  results (`~/foldforge-scratch/paper/F9/h100_kit_partial_user_decision/`, not a limit) and the collector ran without them.
- `data/existing/`: the pre-3.0.6 AF3 limits (H100 3.0.5, A100 3.0.5, RTX 5090 3.0.5 with the n = 1 NaN), kept as the record.

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldForge `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F9 (this folder):** `data/F9_b200.csv` (22 rows = 11 predictors x engine bf16 / PyTorch operators; no FP8 / kit arm) and `data/F9_b200_sizes.csv` (193 probes). Every limit is out-of-memory-bounded (`first_not_ok_status = oom`, none non-monotonic, no non-finite). Largest n (x128 tokens) that completes: engine AF3 56, OpenFold3 56, Boltz2 53, Protenix v1 53, RoseTTAFold3 52, Protenix v2 42, Chai-1 39, ESMFold2 39, ESMFold2-Fast 39, IntelliFold2 28, OpenDDE 21; PyTorch operators AF3 12, Boltz2 12, OpenFold3 12, Protenix v1 12, RoseTTAFold3 12, Chai-1 11, ESMFold2 27, ESMFold2-Fast 27, Protenix v2 10, IntelliFold2 9, OpenDDE 4. Engine label B.
**Chai-1 engine, n = 40:** CUDA illegal memory access after near-full reserved memory (176.9 / 178.35 GiB), classified as OOM (`run_size.py`: a non-OOM `CUDA error` with the process peak reserved >= 98 % of the card is `oom`, the original error is kept in the sizes record field `oom_classified_from`); the same cell under `CUDA_LAUNCH_BLOCKING` gives a clean CUDA out of memory (diagnosis of the engine session). The only cell affected by this rule on this card. The first record (n = 40 `failed`, illegal memory access) is kept in `paper/invalid_archive/f9_chai1_engine_n40_ima/` on the B200. Chai-1 edge: 39 ok / 40 OOM.

## RTX 5090: engine 3.0.10 arm (measured on branch 5090-int), random weights on a rented 4x RTX 5090 (DATA_STATUS, 2026-11-24)

**Released as 3.0.10 (2026-11-24):** engine `5090-int` 07bf36ae4 -> v3.0.10 changes no code the RTX 5090 runs (TokenDiTMiniWorldBiasOnly `backend_versions` number, the GPU-less prebuild plan, an A100-only `MWE_SM120` guard in `wide_ops.cu`), so these rows are the 3.0.10 rows; the CSV `engine` columns say so and keep the measured commit.

`data/F9_rtx5090_3010.csv` (9 rows, engine bf16) and `data/F9_rtx5090_sizes_3010.csv` (every probe). `data/F9_rtx5090.csv`
(released engine 3.0.6, real weights, cssb-rtx) is unchanged; the new CSVs carry the previous value per row (`largest_ok_prev`, `first_not_ok_prev`,
`peak_alloc_gib_at_edge_prev`, `edge_change`) and per probe (`status_prev`, `model_peak_alloc_gib_prev`, `peak_ratio`).

- **Engine:** branch `5090-int` 07bf36ae4 as source (`ENGINE_SRC`; 3.0.9 + origin/main + Boltz-2 up-gate transition / token pair-bias hoist on sm_120 +
  FP8 routes, FP8 off). **Unreleased.**
- **Machine and weights:** a rented 4x RTX 5090 node (`vast5090`, 32 GB each, 575 W, 503 GB host RAM, driver 595.99.02), FoldForge 49d41bb + the F9 helpers
  of this commit range. **Random weights in process** (`FF_PAPER_RANDOM_INIT=1`, `ffpaper.install_random_init`): no checkpoint is read (no weights were
  uploaded to the rented machine; Chai-1's ESM2 input tower, which is not a model parameter, is read from the checkpoint directory). Why this is valid: F9
  measures whether one forward fits in GPU memory and its peak; parameter shapes, dtypes and the code path are those of the real model, the values are
  not. Checked: every edge cell that both runs share has the same peak to the byte as the real-weight 3.0.6 run where the edge phase is unchanged
  (AF3 / OpenFold3 n = 25, Protenix v1 n = 23, RF3 n = 19, OpenDDE n = 4 / 6 / 7). `ok` = the forward completed (`finite` and `random_init` recorded per
  probe; outputs of random weights are not meaningful). The old 5090 F9 rows used the real weights on cssb-rtx.
- **Protocol:** `run_size.py` / `limit_search.py` as before (compile + CUDA graphs + bucketing, one fresh process per probe), one GPU lock per probe
  (`FF_PAPER_PROBE_PREFIX`), the search started AT the previous 5090 limit (`--down`: if it completes, the next size and then the protocol's 2 step
  sizes past the edge; if not, one copy down until a size completes). Some smaller sizes were probed before the start-at-limit change; they are kept.
- **Result:** the largest n is unchanged for 8 of 9 predictors (AF3 25, OpenFold3 25, Protenix v1 23, Boltz-2 20, Protenix v2 19, RoseTTAFold3 19,
  Chai-1 11, OpenDDE 7); **IntelliFold2 10 -> 9** (n = 10, 1,280 tokens, is now out of memory; 3.0.6: 29.20 GiB at n = 10; the rented cards report 31.36 GiB, cssb-rtx 31.33 GiB, so the card is not
  smaller). Every first
  failing size is an OOM; no non-monotonic edge. Peak at the edge: same bytes for AF3 / OpenFold3 / Protenix v1 / RF3 / OpenDDE (the edge peak is in a
  phase the branch did not change), Boltz-2 +1.2 % (29.03 GiB), Protenix v2 +0.03 %, Chai-1 +8.5 % (26.84 -> 29.13 GiB). Below the edge the peak rose
  more where the diffusion phase dominates (AF3 n = 16 +19 %, Boltz-2 n = 8 / 16 +16 / +17 %), as in F3.
