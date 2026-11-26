# F3: 5I28, 1-16 copies, seconds per forward: Anthropic kits vs FoldJet bf16 / FP8

Paper: Fig 1e, Supp 9, Table 6. Cards: H100, A100 (cssb3), RTX 5090 (cssb-rtx). FP8 on the H100 only.

## What is measured

Seconds per complete model forward (trunk passes, sampler, confidence heads; CUDA-synchronized; no featurization / loading / writing) of **5I28** (azurin, 128
residues) with n = 1, 2, 4, 8, 12, 16 identical chains = 128 .. 2,048 tokens. Protocol: `docs/guides/BENCHMARK-5I28.md` (real MSA `P0113359`, 4 distinct real
templates, 4 trunk passes, 200 steps, 5 samples, bf16; RoseTTAFold3 one template per chain, ESMFold2 none).

- **FoldJet** (`run_ff.py`, harness = `benchmarks/5i28_ff/run_5i28_ff.py` unchanged): 11 predictors `protenix1 protenix2 opendde esmfold2 esmfold2-fast
  intellifold2 af3 boltz2 chai1 openfold3 rosettafold3`; arms `engine` (bf16 kernels, backend miniworld) and `engine_fp8` (H100: `FOLDJET_ENGINE_FP8=safe`,
  `FOLDJET_ENGINE_FP8_MODEL=<predictor>`, lazy guards; `fp8.check()` after every cell -> column `fp8_saturated`, and `fp8_status.json` with the route counters). compile +
  CUDA graphs, `scope: trunk`. One process per (predictor, arm, repetition), sizes ascending, one predictor call per size with `benchmark_repeats` 6 warm forwards.
- **Kits** (`kits_h100.sbatch`, `kits_a100.sbatch`, `kits/`): the 12 kits `af3_jax af3_torch openfold3 rf3 chai1 esmfold2 esmfold2_fast boltz2 protenix_v1
  protenix_v2 opendde intellifold2(_notxla)` through the 2026-11-18 steady harness (`$KIT_BENCH`, outside the repository: the kit installs are large). Kit-native
  modes only: `fast`, and the kit's `big` mode for a cell that ran out of memory under `fast` (fresh process, `fb.sh`); no other retry. The kits are not modified
  (the one exception, the af3_torch featuriser `ref_max_modified_date` line, is already in the harness copy; original `featurise.py.orig_kit`).
  AF3 is compared against both the JAX and the PyTorch kit. RTX 5090: kit rows in `data/F3_kits_rtx5090.csv` (see the RTX 5090 kit section below; host RAM cap 16 GB, several kits refuse sm_120).
- **Steady value:** forward 1 (cold) and forward 2 are dropped, then the median of the rest (kits: 7 forwards -> 5 timed; FoldJet: cold + 6 warm -> the first
  warm one dropped -> 5). Three repetitions, each in a different job; the cell = the median of the three. A cell with any failed repetition is not a number
  (`oom` / `failed` in the CSV, never drawn).
- Same node for kits and FoldJet of one card (`NODE=` on the H100 cluster), same inputs.

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

Full campaign (after the release is pinned and one FoldJet commit is chosen; **not started by this branch**):

```bash
CARD=h100 NODE=node02 benchmarks/paper/F3_5i28_times/submit_all.sh      # 66 FoldJet jobs (11 predictors x bf16 / FP8 x 3 repetitions) + 36 kit jobs
CARD=a100 benchmarks/paper/F3_5i28_times/submit_all.sh                  # on cssb3: 33 FoldJet jobs + kit jobs if KIT_BENCH has the kits
# RTX 5090: no scheduler, a loop under the GPU lock (33 runs):
for rep in 1 2 3; do for m in protenix1 protenix2 opendde esmfold2 esmfold2-fast intellifold2 af3 boltz2 chai1 openfold3 rosettafold3; do
  MODEL=$m ARM=engine REP=$rep benchmarks/paper/F3_5i28_times/rtx5090.sh 7200; done; done
```

Every job is ONE continuous run (no chunking); a refused submit stops `submit_all.sh` (ask, do not change the job shape). Expected wall time per job (smoke-based
estimates, H100): FoldJet 15-45 min (compile per size dominates the small sizes, the n = 16 forward the large ones: OpenDDE 63 s x 7); kits 30 min - 6 h
(Chai-1 is the longest: one n = 16 fold takes ~8 min). A100 about 1.7x, RTX 5090 about 2x and its n >= 13 sizes may be out of memory (recorded).

## Output

Run directories: `~/foldforge-scratch/paper/F3/<card>/{engine,engine_fp8}_rep<r>/<model>/{cells.jsonl,meta.json,gpu_clocks.csv,fp8_status.json}` and
`.../<card>/kits_rep<r>/{queues,done_final}`. Every cell row carries FoldJet commit, engine version / source, GPU, node, Slurm job id.

```bash
python benchmarks/paper/F3_5i28_times/collect.py --card h100        # pure stdlib; also fine as a CPU job
```

- `data/F3_foldjet_<card>.csv`: `run,model,n,status,s` (= the columns of `docs/records/data/5i28_foldjet_*`, run `bf16` | `fp8`) + `card,tokens,reps,s_reps,peak_alloc_gib,fp8_saturated,jobs,foldforge_commit,engine`.
- `data/F3_kits_<card>.csv`: `kit,n,tokens,value_s,mode,status,n_timed,forwards_s,templates,source,job,node` (= `5i28_kits_baseline_20261118.csv`) + `reps,value_reps`.

## Status

See `../README.md`. Smoke results of this branch: see the commit notes / the final report. No full campaign has been run from this branch.

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldJet `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F3 (this folder):** `data/F3_foldjet_b200.csv`, 66 rows = 11 predictors x n 1, 2, 4, 8, 12, 16 (FoldJet engine bf16, 3 repetitions each, steady median; every cell finite: 66 ok, 0 failed, 0 OOM; peak allocated at n = 16: 24 GiB (Protenix v1) to 99 GiB (OpenDDE), far below the 178 GiB card). n = 16 s: AF3 10.89, Boltz2 13.36, Chai-1 17.24, ESMFold2 7.71, ESMFold2-Fast 5.18, IntelliFold2 49.53, OpenDDE 65.63, OpenFold3 11.05, Protenix v1 10.01, Protenix v2 19.26, RoseTTAFold3 10.36. No FP8 arm (B200 TriAttn q / k FP8 is opt-in, off by default) and no Anthropic kit arm on this card. Engine label B (column `engine_label`).
**Boltz2, Chai-1, RoseTTAFold3 were measured with one process per size** (`f3s` tasks: one `run_ff.py` per n): one process over several ascending sizes gives non-finite output on the B200 when a new size is run as a warm CUDA-graph replay after an earlier size's sampler graph is cached (eager / cold / `sampler_graph_cache=False` are fine, reproduces on an idle GPU; unrelated to the TriAttn clamp; cause presumed in the engine's capture / sampler-graph cache interaction, unresolved). The warm-repeat meaning is unchanged, so the cells are comparable. The earlier multi-size records (non-finite cells) are not in the CSV. Other predictors ran all sizes in one process and are finite.

## A100 OpenDDE rows re-measured on engine 3.0.9 (DATA_STATUS, 2026-11-23)

(Superseded by the full A100 re-measure below.) `data/F3_foldjet_a100.csv` at that time: the six OpenDDE rows are now engine 3.0.9 (the 3.0.6 value stays in the new column `engine_3.0.6`; every other row is unchanged, 3.0.6). The 3.0.6 run sent `TokenDiTProtenix` `extra_pair_bias` to Triton; 3.0.8 / 3.0.9 serve it on cuda. Same protocol (6 sizes, 3 repetitions in 3 jobs, steady median), same FoldJet checkout as the F4 OpenDDE rerun (c129a81 = 6a0593e + allow_no_protein), the 3.0.9 release artifacts (`foldjet-engine` 3.0.9 + `kernels-sm80-cu129` 3.0.9, extracted, put in front of the 3.0.6 environment through `ENGINE_SRC`), warm caches kept; Slurm 97587 / 97588 / 97589 (gpu01, three jobs on one node as in the 3.0.6 campaign). Backend report: `all_cuda: true`, `TokenDiTProtenix` 24 modules set cuda, refused none.

| tokens | 3.0.6 (s) | 3.0.9 (s) | 3.0.6 / 3.0.9 | Anthropic kit (s) | kit / FoldJet 3.0.9 |
|---|---|---|---|---|---|
| 128 | 2.640 | 2.077 | 1.27 | 6.747 | 3.25 |
| 256 | 5.901 | 4.267 | 1.38 | 6.896 | 1.62 |
| 512 | 18.002 | 12.249 | 1.47 | 15.657 | 1.28 |
| 1,024 | 60.712 | 44.700 | 1.36 | 57.319 | 1.28 |
| 1,536 | 363.895 | 109.623 | 3.32 | 140.983 | 1.29 |
| 2,048 | 656.458 | 199.630 | 3.29 | 1,038.690 (kit `big` mode after fast OOM) | 5.20 |

The 3.0.6 ratios 0.9 (1,024) and 0.4 (1,536) against the kit are now 1.28 and 1.29. Repetition 1 ran about 5 % slower than 2 and 3 at every size (the median is used).

## A100: all 11 predictors re-measured, ESMFold2 family on the release sampler (134 steps), engine main 10f06f87e (2026-11-23)

`data/F3_foldjet_a100.csv` now holds this run for all 66 cells (it replaces the 3.0.6 campaign and the 3.0.9 OpenDDE rows above; those values stay in git history). The separate `F3_foldjet_a100_esmfold2_hoist_unreleased.csv` (200 steps) is removed.

- **Why the ESMFold2 family was slower than the kit on the A100:** FoldJet ran 200 diffusion steps; the release and the kit run 134 (the 200-level schedule capped at max_inference_sigma = 256 A, FoldJet 5553d19 / 7a43e26). Seen in the profiles as 200 sampler kernels per forward (FoldJet) vs 134 `_kabsch3` / sampler-graph replays per fold (kit). The engine side was connected correctly (token DiT, atom block, trunk all on cuda). The A100 engine patches (pair-bias hoist per fold, ESMFold2 atom-block inference op, token-DiT conditioning pack) add on top; they were not the main cause.
- **Engine:** the 3.0.9 release artifacts with the A100 patches as source (`ENGINE_SRC`), the same code as engine `origin/main` 10f06f87e for every touched file (unreleased; judge owed 0 on a100_sm80 / h100_sm90 / 5090_sm120; the patches are gated to capability 8.0, the RTX 5090 is unchanged). Column `engine_label`.
- **FoldJet:** c129a81 + 7a43e26 (= the 5553d19 fix on main). Protocol unchanged: n 1, 2, 4, 8, 12, 16, 3 repetitions in 3 jobs per model, steady median; Slurm 97631-97640, 97646-97669 (cssb3, A100 80GB PCIe). Every cell ok. Allocator "expandable_segments ... OOM" warnings near n = 16 for OpenDDE / IntelliFold2 are retries; the cells finished.

Kit time / FoldJet engine time (> 1 = FoldJet faster; kit = the 2026-11-18 A100 kit baseline, best of af3_jax / af3_torch for AF3, best of the two IntelliFold-v2 runs):

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

FoldJet ran the ESMFold2 family with 200 diffusion steps; the release and the kits run 134 (noise schedule capped at max_inference_sigma = 256 A,
fix 5553d19). Re-run on FoldJet main 49d41bb, engine 3.0.9, node02, engine and engine FP8, 3 repetitions each (Slurm 65809-65820), all sizes ok,
`all_cuda: true` in every process: `data/F3_foldjet_h100_esmfold2_134.csv` replaces the ESMFold2 rows of `data/existing/F3_h100.csv`. Kits unchanged
(they already ran 134 steps).

| model, n | engine bf16 (s): 200 st. / 3.0.1 -> 134 st. / 3.0.9 | engine FP8 | kit (s) |
|---|---|---|---|
| ESMFold2, n=1 | 0.238 -> 0.186 | 0.228 -> 0.176 | 0.308 |
| ESMFold2, n=8 | 3.171 -> 2.870 | 2.713 -> 2.438 | 3.944 |
| ESMFold2, n=16 | 13.347 -> 12.322 | 11.128 -> 9.935 | OOM |
| ESMFold2-Fast, n=1 | 0.217 -> 0.165 | 0.209 -> 0.156 | 0.284 |
| ESMFold2-Fast, n=8 | 2.122 -> 1.825 | 1.871 -> 1.596 | 2.780 |
| ESMFold2-Fast, n=16 | 8.609 -> 7.840 | 7.597 -> 6.510 | OOM |


## RTX 5090: all 9 predictors on the unreleased boltz2-5090 engine (DATA_STATUS, 2026-11-23)

`data/F3_foldjet_rtx5090_boltz2fix_unreleased.csv` (54 rows). `data/F3_foldjet_rtx5090.csv` (released engine 3.0.6) is unchanged; the new CSV carries the 3.0.6 value next to each cell (`s_306`, `s306_over_s`) and the kit value (`kit_s`, `kit_over_s`, from `data/F3_kits_rtx5090.csv`).

- **Why:** on the 5090 FoldJet Boltz-2 was slower than the Anthropic kit at n = 1 / 2 (1.416 / 2.132 s vs 1.123 / 2.047 s). Profile (n = 1 warm forward): no Triton / torch fallback, no host gap; the extra time was the diffusion token DiT: (1) the Boltz-2 up-gated transition ran as the module's own torch code on sm_120 (fp32 master -> bf16 weight casts per call + split-K GEMMs, ~0.55 s per forward); (2) the step-invariant token pair bias was recomputed every sampler step (4,800 `pair_bias_gen` launches) because the A100 hoist was gated to capability 8.0.
- **Engine:** 3.0.9 release artifacts + engine branch `boltz2-5090` as source (`ENGINE_SRC`): a77511b5a (Boltz-2 up-gate transition on `conditioned_transition_sm80` at sm_120 inference) and aa52c4f1a (token pair-bias hoist for capability (12, 0); token DiT `cuda:5090_sm120` bumped, judge owed 0). **Unreleased.** Column `engine`.
- **FoldJet:** `~/ff/FoldJet` 49d41bb on cssb-rtx, `rtx5090.sh` protocol unchanged (n 1, 2, 4, 8, 12, 16; 3 repetitions in separate processes; steady median), every run under the GPU lock, 2026-11-23 13:49-18:56 KST. Every cell `all_cuda`.
- **Result:** every measured cell is faster than 3.0.6: Boltz-2 1.11-1.34x, OpenDDE 1.40-1.63x (the hoist), the other 7 predictors 1.06-1.11x. Repetitions agree within 0.6 %. The OOM cells are the same as on 3.0.6 (Chai-1 / IntelliFold2 n = 12, 16; OpenDDE n = 8, 12, 16; Protenix v2 n = 16); no new OOM. Peak allocated memory rises by 2-15 % (cached bf16 weight copies, hoisted pair bias; e.g. Boltz-2 n = 8 7.22 -> 8.31 GiB, RoseTTAFold3 n = 16 23.20 -> 26.24 GiB).

3.0.6 time / new time:

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| AF3 | 1.07 | 1.06 | 1.09 | 1.11 | 1.10 | 1.09 |
| Boltz-2 | 1.34 | 1.21 | 1.15 | 1.15 | 1.14 | 1.11 |
| Chai-1 | 1.08 | 1.06 | 1.08 | 1.07 | OOM | OOM |
| IntelliFold2 | 1.09 | 1.09 | 1.09 | 1.07 | OOM | OOM |
| OpenDDE | 1.40 | 1.42 | 1.63 | OOM | OOM | OOM |
| OpenFold3 | 1.06 | 1.06 | 1.09 | 1.11 | 1.10 | 1.09 |
| Protenix v1 | 1.06 | 1.06 | 1.09 | 1.11 | 1.10 | 1.09 |
| Protenix v2 | 1.09 | 1.08 | 1.11 | 1.11 | 1.09 | OOM |
| RoseTTAFold3 | 1.06 | 1.06 | 1.09 | 1.11 | 1.10 | 1.09 |

Kit time / new FoldJet time (> 1 = FoldJet faster; cells without a kit value: refused on sm_120, 16 GB host RAM cap or OOM on the kit side):

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| AF3 (af3_torch) | 1.29 | 1.18 | 1.37 | 1.39 | 1.29 | 1.26 |
| Boltz-2 | 1.06 | 1.16 | 1.63 (kit big) | 1.49 (kit big) | – | – |
| Chai-1 | 6.78 | 3.28 | 2.89 | – | – | – |
| OpenFold3 (kit exact, 4 queries) | 2.37 | 2.20 | – | – | – | – |
| Protenix v1 | 2.07 | 1.85 | 1.87 | 1.71 | – | – |
| RoseTTAFold3 | 2.08 | 2.20 | 1.66 | – | – | – |

Every comparable cell is faster than the kit (minimum 1.06, Boltz-2 n = 1; on 3.0.6 Boltz-2 n = 1 / 2 were 0.79 / 0.96). These rows replace `F3_foldjet_rtx5090.csv` only after the engine change is released.

## RTX 5090: Anthropic kit arm (DATA_STATUS, 2026-11-23)

`data/F3_kits_rtx5090.csv` (78 rows = 13 kit runs x n 1, 2, 4, 8, 12, 16; `collect_kits_rtx5090.py`). Host cssb-rtx (RTX 5090 32 GB, sm_120; every process in `mwe.slice`,
**16 GB host RAM in total**, under the GPU lock `~/mwe/gpu.sh`; no Slurm). Kit installs and harness: `~/miniworld-engine-scratch/bench5i28_5090` +
`~/foldforge-scratch/anthropic-kits` on cssb-rtx (outside the repository), kits not modified, kit-native modes (`fast`, `big` after a fast OOM; OpenFold3:
see below). Three repetitions, each its own process per cell: repetition 1 = the 2026-11-18 serial campaign (`data/existing/kits_rtx5090_20261118.csv`),
repetitions 2 and 3 = `run_rep.sh` on 2026-11-23 (10:03-11:24 and 11:24-13:37 KST, cooldown between cells; records `~/mwe/agents/f3_kit/reps/rep{2,3}`).
Only the cells that produced a number in repetition 1 were repeated; a deterministic failure of repetition 1 (RAM cap at a smaller n / refused on sm_120 /
GPU OOM in every mode) keeps its recorded reason. Steady = forwards 3..7 median per repetition; cell = median of the three; any failed repetition -> not a number.

| kit | n = 1 | 2 | 4 | 8 | 12 | 16 | status on the RTX 5090 |
|---|---|---|---|---|---|---|---|
| af3_torch | 1.091 | 1.685 | 4.071 | 12.537 | 27.607 | 48.998 (big) | runs; kit partial (exit 3: fpf trimul_exact / opm / glu_proj levers not served on sm_120, stock fallback) |
| af3_jax | - | - | - | - | - | - | refused on sm_120 (Pallas serve layer: no tile table for cc 12.0; fast / big exit 3) |
| boltz2 | 1.123 | 2.047 | 6.231 (big) | 17.428 (big, 4 timed) | RAM cap | RAM cap | runs, kit levers not active on sm_120 (exit 3); n >= 12 killed by the 16 GB host RAM cap |
| chai1 | 4.477 | 4.808 | 11.413 | RAM cap | RAM cap | RAM cap | runs (kit exit 1 after 7 finished forwards: PAIRTRACK gate refused, templates present); n >= 8 RAM cap |
| esmfold2 | RAM cap | RAM cap | RAM cap | RAM cap | RAM cap | RAM cap | not feasible: 16 GB host RAM cap (weights) |
| esmfold2_fast | RAM cap | RAM cap | RAM cap | RAM cap | RAM cap | RAM cap | not feasible: killed by systemd-oomd while loading the 24.4 GiB checkpoint (probe n = 1, 2026-11-23) |
| intellifold2 (+ notxla) | - | - | - | - | - | - | refused on sm_120 (af3-jax-opt core, FPF_TRIATT / FPF_TRIMUL no_tiles_cc120, 0/35 predictions; probes n = 1, 2026-11-23) |
| opendde | - | - | - | - | - | - | refused on sm_120 (every mode) |
| openfold3 | 2.011 (exact, 4 queries) | 3.146 (exact, 4 queries) | RAM cap | RAM cap | RAM cap | RAM cap | fast refused on sm_120 (pair_transition core); 7 queries exceed 16 GB RAM; `exact` with 4 queries (2 timed) at n = 1, 2; n = 2 repetition 2 was first killed by the RAM cap (rc 137) and re-run as its own process (reps 3.145 / 3.153 / 3.146 s) |
| protenix_v1 | 1.699 | 2.571 | 5.439 | 15.029 | GPU OOM | GPU OOM | runs; n >= 12 out of GPU memory in every mode |
| protenix_v2 | - | - | - | - | - | - | refused on sm_120 (fast / big / exact / off) |
| rosettafold3 | 1.928 | 3.333 | 5.198 | RAM cap | RAM cap | RAM cap | runs; n >= 8 RAM cap (fast and big) |

Kit time / FoldJet engine bf16 time on the same card (`data/F3_foldjet_rtx5090.csv`; > 1 = FoldJet faster):

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| AF3 (af3_torch) | 1.21 | 1.12 | 1.25 | 1.26 | 1.18 | 1.16 |
| Boltz-2 | 0.79 | 0.96 | 1.42 | 1.29 | kit RAM cap | kit RAM cap |
| Chai-1 | 6.27 | 3.08 | 2.67 | kit RAM cap | both OOM / RAM cap | both OOM / RAM cap |
| OpenFold3 | 2.23 (kit exact) | kit failed rep. | kit RAM cap | kit RAM cap | kit RAM cap | kit RAM cap |
| Protenix v1 | 1.96 | 1.75 | 1.71 | 1.54 | kit GPU OOM | kit GPU OOM |
| RoseTTAFold3 | 1.96 | 2.08 | 1.51 | kit RAM cap | kit RAM cap | kit RAM cap |

Boltz-2 n = 1 / 2: the kit is faster on this card (its levers are not active on sm_120, so this is the kit's stock path). ESMFold2 / ESMFold2-Fast: no
comparison on this card (kit and FoldJet both not feasible under the 16 GB host RAM cap); note that the FoldJet ESMFold2-family rows on every card
measured before fix 7a43e26 ran 200 sampler steps (release / kit: 134) and must not be compared with kit rows as they are.

## B200: engine arm re-measured on engine main a553001e1, Anthropic kit arm, ESMFold2 at the release sampler (DATA_STATUS, 2026-11-23)

Shared B200 host (DCTN-0831174147-1, 8 x B200 sm_100), one task per GPU, each pinned to its GPU slot's 9 CPUs (72 / 8), no foreign process on a timed GPU.

- **Engine arm** (`data/F3_foldjet_b200.csv`, replaces the 2026-11-22 label-B rows; the old value of every cell is kept in the column
  `s_campaign_20261122_labelB_200steps_for_esmfold2`): engine main `a553001e1` (source, sm100a prebuilt JIT cache; contains the B200 atom-local
  tail on the tensor cores, the CUDA-graph `record_stream` fix and the B200 LayerNorm D64 / D128 bf16 CUDA route), same protocol, 33 (model, rep)
  processes all rc 0, 66 cells ok. Boltz-2 / Chai-1 / RoseTTAFold3 still one process per size: a multi-size check of Boltz-2 in one process is
  still non-finite on the B200 (rc 3) after the `record_stream` fix. Backend: all classes cuda in every first-size table (Triton 0), except the
  declared B200 LayerNorm exception (width 256 bf16, large row counts → Triton) in Chai-1 (n >= 4) and ESMFold2 / -Fast (n >= 4).
  ESMFold2 / ESMFold2-Fast rows are the release sampler (134 steps; FoldJet ed6efa8b3 + 22febf7 + 5553d19 as a working-tree patch).
  n = 16 (s): AF3 10.61 (10.89 before), Boltz-2 13.57 (13.36), Chai-1 17.15 (17.24), IntelliFold2 51.45 (49.53), OpenDDE 40.98 (65.63),
  OpenFold3 10.98 (11.05), Protenix v1 10.05 (10.01), Protenix v2 19.71 (19.26), RoseTTAFold3 10.40 (10.36), ESMFold2 7.32 (7.71 at 200 steps),
  ESMFold2-Fast 4.77 (5.18 at 200 steps).
- **Anthropic kit arm** (`data/F3_kits_b200.csv`, report `data/F3_kits_b200_REPORT.md`): kit checkout f4f62fa6 installed fresh for sm_100 (Python
  venv route), mode fast (big after a fast OOM), steady value = drop forwards 1-2, median of the rest; cell = median of 3 repetitions (separate
  processes). **Random weights in process** (user order; loaders intercepted at run time, value-free checkpoint skeletons only; no kit file edited).
  Only openfold3 / openfold3_ob0 / protenix_v2 ship a B200 config. n = 16 (s): Boltz-2 24.79, Protenix v1 33.39, Protenix v2 59.02,
  af3_torch 21.74, af3_jax 36.04 (kit exit 5: dpa kernels refused, all forwards run), RF3 28.76, Chai-1 56.28 (exit 1 'PAIRTRACK GATE REFUSED',
  as on the H100), IntelliFold2-notxla 106.37 (exit 3 partial), IntelliFold2 out of memory (fast + big). Not runnable on sm_100 (kit verdicts):
  OpenDDE (cuEquivariance 0.10 without sm_100 code: KERNELS REFUSED), OpenFold3 (pair_transition 'arch-sm100-not-measured' with its own b200.env),
  ESMFold2 / -Fast (Triton OutOfResources: tensor memory 528 > 512 in the kit's sm90-tuned kernel).
  Deviations recorded per cell: CPU pinning from 07:35 UTC (earlier items unpinned on a loaded host; timed forwards agree across reps), RF3 n1-n8 with
  the kit's own switch MODEL_OPT_LEVERS_OFF=prefetch (its niced prefetch helper starved; featurisation is outside the timed forward).
- **Engine vs kit, same host:** the engine is faster in all 47 comparable cells (n = 1..16); the smallest margin is Boltz-2 n = 1 (1.70x; n = 16: 1.83x);
  n = 16: AF3 2.05x (af3_torch) / 3.40x (af3_jax), Chai-1 3.28x, Protenix v1 3.32x, Protenix v2 3.00x, RF3 2.77x, IntelliFold2 2.07x (notxla).

## OpenFold3 preview 2 Anthropic kit (2026-11-27)

The paper said "OpenFold3 preview 2 has no Anthropic kit": wrong. The `openfold3` kit of github.com/anthropics/uplifting-biomolecular-modeling (f4f62fa) is
OpenFold3 0.4.1 with the preview-2 weights (`of3-p2-155k.pt`); the kit measured as "openfold3" so far is `openfold3_ob0` (OpenFold3 0.5.0, OpenBind-0).
Data: `data/F3_kits_openfold3_p2.csv` (kit `openfold3_p2`; same columns idea as the other kit CSVs: steady value = median of the 3 repetitions, each the
median of 5 forwards after forward 1 and 2 are dropped; `forwards_s` = all 7 forwards per repetition, `|`-separated).

- **FoldJet side is the existing `openfold3` rows**: FoldJet runs `openfold3` and `openfold3-preview2` through the same `AlphaFold3` class and settings, only the
  weights differ (H100 n = 1 check: preview2 0.403 s, openfold3 0.39 s).
- **Kit setup** (`kits/openfold3_p2/`, copies of the harness files; the kit itself is unmodified, installed by its STOCK.md route C): 0.4.1 has no CIF-direct
  template fields, so the 4 distinct real templates go in as a template alignment (`templates_4.a3m`: query + full chain sequences of 5SYD_A, 6KOL_A, 5YT7_A,
  6TFD_A, realigned by the kit's own kalign path; the 20-hit a3m drops insertions, and 0.4.1 then dropped 6KOL / 6TFD); `--use-templates true --upstream-fix
  OF3-001` (the kit's own fix: stock 0.4.1 preprocesses templates but never hands them to the model); every cell logs `TEMPLATES FEATURISED real_slots=4/4`,
  MSA 6,790 rows, 4 trunk passes, 200 steps, 5 samples, mode fast (no cell needed big). H100 / A100: in the kits' Ubuntu 22.04 container with the CUDA 13
  compat libcuda (`cu130kit.sh`, as `openfold3_ob0`): natively on the 575 driver the kit's TriMul `tma` Triton kernel segfaults from n = 4.
- **Runs**: H100 node01 job 67620, A100 gpu03 job 100100 (`of3p2_ngpu.sh`: one job, 4 GPUs, one kit process per GPU at a time, repetitions 1-3 interleaved).

| n | H100 kit (s) | kit / FoldJet bf16 | kit / FoldJet FP8 | A100 kit (s) | kit / FoldJet bf16 |
|---|---|---|---|---|---|
| 1 | 0.868 | 2.24 | 2.28 | 1.413 | 1.33 |
| 2 | 1.154 | 1.95 | 2.00 | 2.227 | 1.26 |
| 4 | 2.184 | 1.74 | 1.88 | failed | - |
| 8 | 5.891 | 1.56 | 1.82 | 12.210 | 1.21 |
| 12 | 12.818 | 1.44 | 1.75 | 27.081 | 1.16 |
| 16 | 23.686 | 1.40 | 1.78 | 50.510 | 1.16 |

FoldJet values: `F3_foldforge_h100_3010.csv` (engine 3.0.10) and `F3_foldforge_a100.csv` (engine 3.0.9 + A100 patches), `openfold3` rows.
- **A100 n = 4**: all 3 repetitions died in the 3rd forward with `CUDA error: an illegal memory access` (kit exit 1, not out of memory): `failed`.
- **RTX 5090**: fast refused at n = 1 (`pair_transition` core `arch-sm120-not-measured`, strict; the same refusal as `openfold3_ob0`); n = 2 / 4 were killed by
  the 16 GB host RAM cap before reaching that point; exact was not measured (user decision 2026-11-27).
- **B200** (rented vast.ai, random in-process weights): fast refused at n = 1 (`arch-sm100-not-measured`), as `openfold3_ob0` on the B200.
