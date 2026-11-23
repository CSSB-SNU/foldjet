# F3: 5I28, 1-16 copies, seconds per forward: Anthropic kits vs FoldForge bf16 / FP8

Paper: Fig 1e, Supp 9, Table 6. Cards: H100, A100 (cssb3), RTX 5090 (cssb-rtx). FP8 on the H100 only.

## What is measured

Seconds per complete model forward (trunk passes, sampler, confidence heads; CUDA-synchronized; no featurization / loading / writing) of **5I28** (azurin, 128
residues) with n = 1, 2, 4, 8, 12, 16 identical chains = 128 .. 2,048 tokens. Protocol: `docs/guides/BENCHMARK-5I28.md` (real MSA `P0113359`, 4 distinct real
templates, 4 trunk passes, 200 steps, 5 samples, bf16; RoseTTAFold3 one template per chain, ESMFold2 none).

- **FoldForge** (`run_ff.py`, harness = `benchmarks/5i28_ff/run_5i28_ff.py` unchanged): 11 predictors `protenix1 protenix2 opendde esmfold2 esmfold2-fast
  intellifold2 af3 boltz2 chai1 openfold3 rosettafold3`; arms `engine` (bf16 kernels, backend miniworld) and `engine_fp8` (H100: `MINIWORLD_FP8=safe`,
  `MINIWORLD_FP8_MODEL=<predictor>`, lazy guards; `fp8.check()` after every cell -> column `fp8_saturated`, and `fp8_status.json` with the route counters). compile +
  CUDA graphs, `scope: trunk`. One process per (predictor, arm, repetition), sizes ascending, one predictor call per size with `benchmark_repeats` 6 warm forwards.
- **Kits** (`kits_h100.sbatch`, `kits_a100.sbatch`, `kits/`): the 12 kits `af3_jax af3_torch openfold3 rf3 chai1 esmfold2 esmfold2_fast boltz2 protenix_v1
  protenix_v2 opendde intellifold2(_notxla)` through the 2026-11-18 steady harness (`$KIT_BENCH`, outside the repository: the kit installs are large). Kit-native
  modes only: `fast`, and the kit's `big` mode for a cell that ran out of memory under `fast` (fresh process, `fb.sh`); no other retry. The kits are not modified
  (the one exception, the af3_torch featuriser `ref_max_modified_date` line, is already in the harness copy; original `featurise.py.orig_kit`).
  AF3 is compared against both the JAX and the PyTorch kit. **No kit install exists on the RTX 5090**: that card has the FoldForge side only.
- **Steady value:** forward 1 (cold) and forward 2 are dropped, then the median of the rest (kits: 7 forwards -> 5 timed; FoldForge: cold + 6 warm -> the first
  warm one dropped -> 5). Three repetitions, each in a different job; the cell = the median of the three. A cell with any failed repetition is not a number
  (`oom` / `failed` in the CSV, never drawn).
- Same node for kits and FoldForge of one card (`NODE=` on the H100 cluster), same inputs.

## Commands

Smoke (any card; engine candidate tree through `ENGINE_SRC`):

```bash
# H100
ENGINE_SRC=~/miniworld-engine-scratch/rc306/src sbatch --time=00:10:00 -J f3s --export=ALL,MODEL=af3,ARM=engine,REP=1,SIZES="1 2",REPEATS=3 benchmarks/paper/F3_5i28_times/h100.sbatch
# A100 (on cssb3, from ~/practice/paper_infer)
ENGINE_SRC=~/miniworld-engine-scratch/rc306/src sbatch --time=00:10:00 -J f3s --export=ALL,MODEL=af3,ARM=engine,REP=1,SIZES="1 2",REPEATS=3 benchmarks/paper/F3_5i28_times/a100.sbatch
# RTX 5090 (on cssb-rtx, from ~/ff/paper_infer; under the GPU lock)
ENGINE_SRC=~/mwe/rc306/src MODEL=af3 ARM=engine REP=1 SIZES="1 2" REPEATS=3 benchmarks/paper/F3_5i28_times/rtx5090.sh 1500
```

Full campaign (after the release is pinned and one FoldForge commit is chosen; **not started by this branch**):

```bash
CARD=h100 NODE=node02 benchmarks/paper/F3_5i28_times/submit_all.sh      # 66 FoldForge jobs (11 predictors x bf16 / FP8 x 3 repetitions) + 36 kit jobs
CARD=a100 benchmarks/paper/F3_5i28_times/submit_all.sh                  # on cssb3: 33 FoldForge jobs + kit jobs if KIT_BENCH has the kits
# RTX 5090: no scheduler, a loop under the GPU lock (33 runs):
for rep in 1 2 3; do for m in protenix1 protenix2 opendde esmfold2 esmfold2-fast intellifold2 af3 boltz2 chai1 openfold3 rosettafold3; do
  MODEL=$m ARM=engine REP=$rep benchmarks/paper/F3_5i28_times/rtx5090.sh 7200; done; done
```

Every job is ONE continuous run (no chunking); a refused submit stops `submit_all.sh` (ask, do not change the job shape). Expected wall time per job (smoke-based
estimates, H100): FoldForge 15-45 min (compile per size dominates the small sizes, the n = 16 forward the large ones: OpenDDE 63 s x 7); kits 30 min - 6 h
(Chai-1 is the longest: one n = 16 fold takes ~8 min). A100 about 1.7x, RTX 5090 about 2x and its n >= 13 sizes may be out of memory (recorded).

## Output

Run directories: `~/foldforge-scratch/paper/F3/<card>/{engine,engine_fp8}_rep<r>/<model>/{cells.jsonl,meta.json,gpu_clocks.csv,fp8_status.json}` and
`.../<card>/kits_rep<r>/{queues,done_final}`. Every cell row carries FoldForge commit, engine version / source, GPU, node, Slurm job id.

```bash
python benchmarks/paper/F3_5i28_times/collect.py --card h100        # pure stdlib; also fine as a CPU job
```

- `data/F3_foldforge_<card>.csv`: `run,model,n,status,s` (= the columns of `docs/records/data/5i28_foldforge_*`, run `bf16` | `fp8`) + `card,tokens,reps,s_reps,peak_alloc_gib,fp8_saturated,jobs,foldforge_commit,engine`.
- `data/F3_kits_<card>.csv`: `kit,n,tokens,value_s,mode,status,n_timed,forwards_s,templates,source,job,node` (= `5i28_kits_baseline_20261118.csv`) + `reps,value_reps`.

## Status

See `../README.md`. Smoke results of this branch: see the commit notes / the final report. No full campaign has been run from this branch.

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldForge `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F3 (this folder):** `data/F3_foldforge_b200.csv`, 66 rows = 11 predictors x n 1, 2, 4, 8, 12, 16 (FoldForge engine bf16, 3 repetitions each, steady median; every cell finite: 66 ok, 0 failed, 0 OOM; peak allocated at n = 16: 24 GiB (Protenix v1) to 99 GiB (OpenDDE), far below the 178 GiB card). n = 16 s: AF3 10.89, Boltz2 13.36, Chai-1 17.24, ESMFold2 7.71, ESMFold2-Fast 5.18, IntelliFold2 49.53, OpenDDE 65.63, OpenFold3 11.05, Protenix v1 10.01, Protenix v2 19.26, RoseTTAFold3 10.36. No FP8 arm (B200 TriAttn q / k FP8 is opt-in, off by default) and no Anthropic kit arm on this card. Engine label B (column `engine_label`).
**Boltz2, Chai-1, RoseTTAFold3 were measured with one process per size** (`f3s` tasks: one `run_ff.py` per n): one process over several ascending sizes gives non-finite output on the B200 when a new size is run as a warm CUDA-graph replay after an earlier size's sampler graph is cached (eager / cold / `sampler_graph_cache=False` are fine, reproduces on an idle GPU; unrelated to the TriAttn clamp; cause presumed in the engine's capture / sampler-graph cache interaction, unresolved). The warm-repeat meaning is unchanged, so the cells are comparable. The earlier multi-size records (non-finite cells) are not in the CSV. Other predictors ran all sizes in one process and are finite.

## A100 OpenDDE rows re-measured on engine 3.0.9 (DATA_STATUS, 2026-11-23)

(Superseded by the full A100 re-measure below.) `data/F3_foldforge_a100.csv` at that time: the six OpenDDE rows are now engine 3.0.9 (the 3.0.6 value stays in the new column `engine_3.0.6`; every other row is unchanged, 3.0.6). The 3.0.6 run sent `TokenDiTProtenix` `extra_pair_bias` to Triton; 3.0.8 / 3.0.9 serve it on cuda. Same protocol (6 sizes, 3 repetitions in 3 jobs, steady median), same FoldForge checkout as the F4 OpenDDE rerun (c129a81 = 6a0593e + allow_no_protein), the 3.0.9 release artifacts (`miniworld-engine` 3.0.9 + `kernels-sm80-cu129` 3.0.9, extracted, put in front of the 3.0.6 environment through `ENGINE_SRC`), warm caches kept; Slurm 97587 / 97588 / 97589 (gpu01, three jobs on one node as in the 3.0.6 campaign). Backend report: `all_cuda: true`, `TokenDiTProtenix` 24 modules set cuda, refused none.

| tokens | 3.0.6 (s) | 3.0.9 (s) | 3.0.6 / 3.0.9 | Anthropic kit (s) | kit / FoldForge 3.0.9 |
|---|---|---|---|---|---|
| 128 | 2.640 | 2.077 | 1.27 | 6.747 | 3.25 |
| 256 | 5.901 | 4.267 | 1.38 | 6.896 | 1.62 |
| 512 | 18.002 | 12.249 | 1.47 | 15.657 | 1.28 |
| 1,024 | 60.712 | 44.700 | 1.36 | 57.319 | 1.28 |
| 1,536 | 363.895 | 109.623 | 3.32 | 140.983 | 1.29 |
| 2,048 | 656.458 | 199.630 | 3.29 | 1,038.690 (kit `big` mode after fast OOM) | 5.20 |

The 3.0.6 ratios 0.9 (1,024) and 0.4 (1,536) against the kit are now 1.28 and 1.29. Repetition 1 ran about 5 % slower than 2 and 3 at every size (the median is used).

## A100: all 11 predictors re-measured, ESMFold2 family on the release sampler (134 steps), engine main 10f06f87e (2026-11-23)

`data/F3_foldforge_a100.csv` now holds this run for all 66 cells (it replaces the 3.0.6 campaign and the 3.0.9 OpenDDE rows above; those values stay in git history). The separate `F3_foldforge_a100_esmfold2_hoist_unreleased.csv` (200 steps) is removed.

- **Why the ESMFold2 family was slower than the kit on the A100:** FoldForge ran 200 diffusion steps; the release and the kit run 134 (the 200-level schedule capped at max_inference_sigma = 256 A, FoldForge 5553d19 / 7a43e26). Seen in the profiles as 200 sampler kernels per forward (FoldForge) vs 134 `_kabsch3` / sampler-graph replays per fold (kit). The engine side was connected correctly (token DiT, atom block, trunk all on cuda). The A100 engine patches (pair-bias hoist per fold, ESMFold2 atom-block inference op, token-DiT conditioning pack) add on top; they were not the main cause.
- **Engine:** the 3.0.9 release artifacts with the A100 patches as source (`ENGINE_SRC`), the same code as engine `origin/main` 10f06f87e for every touched file (unreleased; judge owed 0 on a100_sm80 / h100_sm90 / 5090_sm120; the patches are gated to capability 8.0, the RTX 5090 is unchanged). Column `engine_label`.
- **FoldForge:** c129a81 + 7a43e26 (= the 5553d19 fix on main). Protocol unchanged: n 1, 2, 4, 8, 12, 16, 3 repetitions in 3 jobs per model, steady median; Slurm 97631-97640, 97646-97669 (cssb3, A100 80GB PCIe). Every cell ok. Allocator "expandable_segments ... OOM" warnings near n = 16 for OpenDDE / IntelliFold2 are retries; the cells finished.

Kit time / FoldForge engine time (> 1 = FoldForge faster; kit = the 2026-11-18 A100 kit baseline, best of af3_jax / af3_torch for AF3, best of the two IntelliFold-v2 runs):

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| AF3 | 1.53 | 1.25 | 1.24 | 1.30 | 1.23 | 1.25 |
| Boltz-2 | 1.43 | 1.60 | 1.88 | 1.86 | 1.65 | 1.60 |
| Chai-1 | 9.17 | 4.90 | 4.29 | 2.95 | 2.28 | 1.95 |
| OpenFold3 | 1.29 | 1.23 | 1.25 | 1.27 | 1.23 | 1.23 |
| Protenix v1 | 3.43 | 2.70 | 2.59 | 2.41 | 2.20 | 2.35 |
| Protenix v2 | 2.40 | 2.16 | 1.83 | 1.58 | 1.43 | 1.48 |
| OpenDDE | 3.98 | 1.91 | 1.49 | 1.44 | 1.27 | 5.12 |
| RoseTTAFold3 | 2.85 | 2.74 | 1.82 | 1.58 | 1.40 | 1.38 |
| ESMFold2 | 1.32 | 1.32 | 1.27 | 1.21 | 1.20 | kit failed (engine 41.04 s) |
| ESMFold2-Fast | 1.33 | 1.39 | 1.38 | 1.33 | 1.35 | kit failed (engine 23.19 s) |
| IntelliFold2 | kit OOM | kit OOM | kit OOM | kit OOM | 1.72 | 1.74 |

Every comparable cell is faster than the kit (minimum 1.20, ESMFold2 n = 12). Against the 3.0.6 campaign, AF3 n = 2 / 4 moved from 1.03 to 1.25 / 1.24 and OpenFold3 from 0.99-1.05 to 1.23-1.29.

## H100: ESMFold2 / ESMFold2-Fast re-measured with the release sampler (134 steps), engine 3.0.9 (2026-11-23)

FoldForge ran the ESMFold2 family with 200 diffusion steps; the release and the kits run 134 (noise schedule capped at max_inference_sigma = 256 A,
fix 5553d19). Re-run on FoldForge main 49d41bb, engine 3.0.9, node02, engine and engine FP8, 3 repetitions each (Slurm 65809-65820), all sizes ok,
`all_cuda: true` in every process: `data/F3_foldforge_h100_esmfold2_134.csv` replaces the ESMFold2 rows of `data/existing/F3_h100.csv`. Kits unchanged
(they already ran 134 steps).

| model, n | engine bf16 (s): 200 st. / 3.0.1 -> 134 st. / 3.0.9 | engine FP8 | kit (s) |
|---|---|---|---|
| ESMFold2, n=1 | 0.238 -> 0.186 | 0.228 -> 0.176 | 0.308 |
| ESMFold2, n=8 | 3.171 -> 2.870 | 2.713 -> 2.438 | 3.944 |
| ESMFold2, n=16 | 13.347 -> 12.322 | 11.128 -> 9.935 | OOM |
| ESMFold2-Fast, n=1 | 0.217 -> 0.165 | 0.209 -> 0.156 | 0.284 |
| ESMFold2-Fast, n=8 | 2.122 -> 1.825 | 1.871 -> 1.596 | 2.780 |
| ESMFold2-Fast, n=16 | 8.609 -> 7.840 | 7.597 -> 6.510 | OOM |
