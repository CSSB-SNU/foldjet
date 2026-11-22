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
