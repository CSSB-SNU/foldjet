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

`data/F3_foldforge_a100.csv`: the six OpenDDE rows are now engine 3.0.9 (the 3.0.6 value stays in the new column `engine_3.0.6`; every other row is unchanged, 3.0.6). The 3.0.6 run sent `TokenDiTProtenix` `extra_pair_bias` to Triton; 3.0.8 / 3.0.9 serve it on cuda. Same protocol (6 sizes, 3 repetitions in 3 jobs, steady median), same FoldForge checkout as the F4 OpenDDE rerun (c129a81 = 6a0593e + allow_no_protein), the 3.0.9 release artifacts (`miniworld-engine` 3.0.9 + `kernels-sm80-cu129` 3.0.9, extracted, put in front of the 3.0.6 environment through `ENGINE_SRC`), warm caches kept; Slurm 97587 / 97588 / 97589 (gpu01, three jobs on one node as in the 3.0.6 campaign). Backend report: `all_cuda: true`, `TokenDiTProtenix` 24 modules set cuda, refused none.

| tokens | 3.0.6 (s) | 3.0.9 (s) | 3.0.6 / 3.0.9 | Anthropic kit (s) | kit / FoldForge 3.0.9 |
|---|---|---|---|---|---|
| 128 | 2.640 | 2.077 | 1.27 | 6.747 | 3.25 |
| 256 | 5.901 | 4.267 | 1.38 | 6.896 | 1.62 |
| 512 | 18.002 | 12.249 | 1.47 | 15.657 | 1.28 |
| 1,024 | 60.712 | 44.700 | 1.36 | 57.319 | 1.28 |
| 1,536 | 363.895 | 109.623 | 3.32 | 140.983 | 1.29 |
| 2,048 | 656.458 | 199.630 | 3.29 | 1,038.690 (kit `big` mode after fast OOM) | 5.20 |

The 3.0.6 ratios 0.9 (1,024) and 0.4 (1,536) against the kit are now 1.28 and 1.29. Repetition 1 ran about 5 % slower than 2 and 3 at every size (the median is used).

## A100 ESMFold2 / ESMFold2-Fast slower than the Anthropic kit: cause and an unreleased engine patch (2026-11-23)

**Not a data error; an engine-side cost.** On the 3.0.6 / 3.0.9 engine the A100 ESMFold2 forward spent about 13 % in `aa80::pair_bias_gen_kernel` (2,400 calls per forward = 12 token-DiT blocks x 200 sampler steps, 47 us each at 128 tokens: the pair LayerNorm + head projection of a pair that is identical at every step) and 3 % in per-call fp32 -> bf16 casts of the block weights (`aten::copy_` of Wo / Ws, 24 per step), plus 29 ms of per-step weight casts / concatenations in the ESMFold2 atom block. The kit hoists the step-invariant pair bias once per fold (its CHANGES.md: "the diffusion transformer's step-invariant pair bias per fold"). The engine hoisted it on H100 / B200 but the sm_80 attention (`augattn_sm80`) only had an opt-in cache for the atom width. Not the cause: the ESM-C language model (run once per predictor call, outside the timed forward; fallback PyTorch ESM-C without transformer-engine / flash-attn costs 0.10-0.22 s per call at n = 1, not in `steady_s`), the sampler glue (cuda `AlignEuler`), TriAttn / TriMul / Transition (all cuda, `all_cuda: true`).
Evidence: torch profiler (CUDA activity only, CUDA-graph replay, 1 warm forward, n = 1): kernel time 890 ms of an 858 ms wall forward on 3.0.9 (GPU bound), `pair_bias_gen_kernel` 113 ms, bf16 copies 43 ms; after the patch 733 ms of 700 ms (pair bias 0, copies 20 ms).

Engine patch (branch `esm-a100` of the engine repository, **unreleased, not pushed**): `c7511f93e` = (1) `kernels/integrations/augattn_sm80.py`: token-width pair bias hoisted per (pair tensor, norm / bias weights, key mask) through `_capture.hoisted_table` (switch `MINIWORLD_TOKEN_BIAS_HOIST=0`; only when the bias is <= 192 MiB per block), out-projection casts of fp32 masters cached (`param_forms`); `TokenDiTProtenix` / `TokenDiTMiniWorld` `cuda:a100_sm80` 6 -> 7 / 7 -> 8; (2) `kernels/integrations/local_dit_esm.py`: an inference op for the A100 ESMFold2 atom block (no saved activations, casts / packs once per parameter version; switch `MINIWORLD_ESM_A100_INFER=0`); `AtomDiTESMFold2Pair` `cuda:a100_sm80` 1 -> 2. Output bits unchanged: same-process A-B (hoist off / on, ABAB) gives identical coordinates (max abs difference 0) at n = 1, 2, 4; accuracy records of `TokenDiTProtenix`, `TokenDiTMiniWorld`, `AtomDiTESMFold2Pair` on the A100 equal the 3.0.9 records (same errors, same colours); `tests/gpu/test_token_pair_bias_hoist_sm80.py` bitwise, 4 passed. Same-process A-B: n = 1 0.858 -> 0.745 s (hoist), 0.725 -> 0.700 s (atom op); n = 4 3.81 -> 3.36 s.

Re-measured with the F3 protocol (3 repetitions, Slurm 97601-97606), engine label **3.0.9 release artifacts + esm-a100 c7511f93e (unreleased)**; `data/F3_foldforge_a100_esmfold2_hoist_unreleased.csv` (column `engine_3.0.6_s` = the 3.0.6 row, which stays untouched in `F3_foldforge_a100.csv`):

| model | tokens | 3.0.6 (s) | patched (s) | kit (s) | kit / FoldForge 3.0.6 | kit / FoldForge patched |
|---|---|---|---|---|---|---|
| ESMFold2 | 128 | 0.855 | 0.699 | 0.620 | 0.73 | 0.89 |
| | 256 | 1.629 | 1.387 | 1.332 | 0.82 | 0.96 |
| | 512 | 3.827 | 3.264 | 3.362 | 0.88 | 1.03 |
| | 1,024 | 12.496 | 10.586 | 11.473 | 0.92 | 1.08 |
| | 1,536 | 28.800 | 24.509 | 26.710 | 0.93 | 1.09 |
| | 2,048 | 50.838 | 43.715 | kit OOM | | |
| ESMFold2-Fast | 128 | 0.787 | 0.632 | 0.542 | 0.69 | 0.86 |
| | 256 | 1.401 | 1.157 | 1.081 | 0.77 | 0.93 |
| | 512 | 2.973 | 2.417 | 2.440 | 0.82 | 1.01 |
| | 1,024 | 8.952 | 6.992 | 7.607 | 0.85 | 1.09 |
| | 1,536 | 20.177 | 15.697 | 17.368 | 0.86 | 1.11 |
| | 2,048 | 34.788 | 27.003 | kit OOM | | |

Remaining: at 128 and 256 tokens the kit is still 4-14 % faster. The patched n = 1 forward is GPU bound (kernel time 734 ms of 700 ms wall): about 370 ms are cuBLAS bf16 GEMMs of the token DiT (M = 640 rows, 2,400 calls per GEMM shape; the kit fuses them in Triton with the conditioning), 12 ms the per-step windowed atom pair bias (`h_pair_bias_esm`, step-invariant, not yet hoisted on sm_80), 19 ms inductor cast kernels. Not closed.
