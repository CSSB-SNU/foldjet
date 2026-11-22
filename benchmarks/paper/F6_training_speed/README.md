# F6: whole-model training speed (AlphaFold 3 and Protenix v1)

Paper: Fig 1d (seconds per training example, one card). Arms: PyTorch, PyTorch + `torch.compile`, cuEquivariance + `torch.compile`, engine.
Cards: H100, A100, RTX 5090.

## What is measured

`scripts/train_protenix1.py --bench` (FoldForge's own training driver), per model (`protenix1`, `af3`) and arm:

| item | value |
|---|---|
| data | `--bench` in-memory stand-in pool: 8 FoldBench targets (`../train_common/pool_targets.txt`, 268-384 tokens) padded to one shape: **384 tokens, 4,096 atoms, MSA pool 16,384 rows (1,024 drawn per recycle), 4 templates**. The pool is built once per model (`pool_<model>.pt`, H100) and copied to the other cards, so every arm and every card sees identical stand-in labels. |
| recipe | `--recipe protenix` (Protenix v1 fine-tuning recipe: loss weights, Adam, clip 10, EMA), precision `af3_default` (bf16 trunk on fp32 masters, **fp32 diffusion head and fp32 pair-bias logits**: unchanged numerics) |
| step | `--step-graph` (whole micro step as one CUDA graph), `--activation-checkpointing` (the `--bench` default), the 20-step mini-rollout and the confidence head are part of every micro step, 1-4 random recycles (the same plan for every arm) |
| diffusion augmentations | 48. Engine: all 48 in one pass (`--augment-chunk 0`). Other arms: first 48 in one pass; if that runs out of memory (recorded), `24` (24 x 2); the card's ladder continues (`AUG_LADDER_OTHER`; 5090: `0 24 12 8 6 4`). **The value used is a column of the CSV.** |
| schedule | 6 optimizer steps x 8 micro steps; step 0 (graph capture / compilation) is excluded; **seconds per example = median of steps 1-5 / 8** |
| repeats | 3 (`REPEATS`), each repeat a new process |
| backends | engine = `--backend miniworld`; PyTorch = `--backend pytorch`; cuEquivariance = `--backend cueq`; `C` = plus `--compile all` |

Rules kept: measurements use the warm caches of `train_common/env.sh` (no fresh `TRITON_CACHE_DIR`), no Triton in an engine run is accepted silently (`triton_calls` column: the
engine dispatch table of the run; the known `Transition` fp32-TF32 refusals are served by Triton and counted there), OOM is a recorded result and is never plotted.

## Files

- `run_f6.sh`: the whole matrix on one card (env `MODELS`, `ARMS`, `REPEATS`, `AUG_LADDER_*`, `ENGINE_SRC`, `SMOKE=1`).
- `h100.sbatch`, `a100.sbatch`, `rtx5090.sh`: per-card launchers. `../train_common/env.sh`: per-card environment (python of the pixi env called directly; never `pixi run`).
- `collect.py`: logs -> `f6_runs_<card>.csv` (one row per process), `f6_summary_<card>.csv` (per model x arm), `steps_<card>.txt` (the shape of paper `data/train_speed/steps.txt`).
  Paper `fig1/make_fig1_panels.py: panel_train` reads `ts48-` lines for the engine and `ts-` lines (24 x 2) for the other arms; if an arm fitted all 48 on a card its line is `ts48-`
  and the paper script has to read it as that arm's value.

## Commands (full campaign; started by the coordinator after the release and the FoldForge pin)

```
# H100 (cluster; one 1-GPU job, ~3 h)
mkdir -p ~/foldforge-scratch/paper/F6_training_speed/h100/pool && cp ~/foldforge-scratch/train_speed/pool_*.pt ~/foldforge-scratch/paper/F6_training_speed/h100/pool/
sbatch benchmarks/paper/F6_training_speed/h100.sbatch                      # ENGINE_SRC=<tree> only for a release candidate
# A100 (ssh cssb3; FoldForge clone at ~/practice/paper-train; pools copied to ~/foldforge-scratch/paper/F6_training_speed/a100/pool)
sbatch benchmarks/paper/F6_training_speed/a100.sbatch
# RTX 5090 (ssh cssb-rtx; clone at ~/mwe/paper-train; pools in ~/ff/paper/F6_training_speed/rtx5090/pool)
nohup ~/mwe/gpu.sh 90000 benchmarks/paper/F6_training_speed/rtx5090.sh > ~/ff/paper/logs/f6.log 2>&1 &
# collect (any machine with python3; not on a login node of the cluster): 
python3 benchmarks/paper/F6_training_speed/collect.py --out-dir ~/foldforge-scratch/paper/F6_training_speed/<card>
```

Record kept with every run: FoldForge commit, engine version and file (BENCH line, `engine_file` shows rc source vs installed wheel), GPU, node, Slurm job id (`meta_<job>.txt`, RUNMETA lines).
Smoke (plumbing only): `SMOKE=1 MODELS=protenix1 ARMS="miniworld pytorchC" ENGINE_SRC=<engine source> sbatch --time=00:10:00 --export=ALL h100.sbatch`.

## Output

`~/foldforge-scratch/paper/F6_training_speed/<card>/` (logs, meta, CSVs). The repository gets only the final CSVs (`benchmarks/paper/F6_training_speed/results/` if wanted).

## H100 engine rows are not engine-CUDA-only (2026-11-21)

In the engine 3.0.1 runs the four fp32 conditioning transitions of the diffusion head (`diffusion_head.{pair,single}_transition_{0,1}`, af3_default precision,
TF32 off) were refused by the engine's H100 cuda backend and served by Triton (48 calls with chunk 0, 72 with chunk 24). The rows carry
`engine_cuda_only = no`; they are not re-measured (details: `~/foldforge-scratch/paper/engine_issues/h100_fp32_transition_triton.md`; engine fix: branch
`h100-transition-fp32-train`).

## H100 engine arm re-measured with the fp32 Transition fix (2026-11-21)

Engine label for the paper table: **engine 3.0.7** (released 2026-11-22, tag v3.0.7 = 09615f07e). Measured on engine main d967378a2, whose `src/` differs from
v3.0.7 only by two lines in `modules/norm.py` (`_wants_grad` skips None tensors: the bias=None BiasOnlyAttention Triton path), which the Transition and the F6
training paths (AF3 / Protenix v1) do not reach; the rest of the difference is the version, the kernels pin, notes and tests. The measurement used the 3.0.6
`_prebuilt` tree, where `ieee_f32_sm90` (the fp32 Transition chain) was JIT-built; the 3.0.7 kernels wheel ships it prebuilt (the same source). Not re-measured
on the installed 3.0.7. Source of the measurement: `git archive d967378a2 src/miniworld_engine`
plus the 3.0.6 `_prebuilt` tree, put in front of the installed engine through `ENGINE_SRC` (the BENCH line says `engine_version 3.0.6` because the package
metadata is the 3.0.6 one; the `engine_file` is the overlay). Same protocol as above (2 models x 3 repeats, 6 optimizer steps x 8 micro steps, 48 augmentations in
one pass, step graph, activation checkpointing, the original H100 stand-in pools), FoldForge worktree `paper-f6-h100` (4209c51, clean). Slurm 64927 (Protenix v1),
64928 (AF3), after a warm-up job (SMOKE=1) for the shared extension cache. The other arms (PyTorch, PyTorch+compile, cuEq(+compile)) are NOT re-run: their rows
stay the engine-3.0.1 H100 values in `data/existing/`; the engine row is therefore from a different session than the other arms.

| model | engine s/example (median of 3) | min | max | peak GiB | Triton calls | refused | 3.0.1 row (engine_cuda_only = no) |
|---|---|---|---|---|---|---|---|
| AF3 | 0.8121 | 0.8033 | 0.8127 | 63.0 | 0 | none | 0.8066 |
| Protenix v1 | 0.7870 | 0.7794 | 0.7874 | 63.0 | 0 | none | 0.7843 |

`engine_cuda_only = yes`: `ENGINE_AFTER_MICRO1` (data/engine_after_micro1_h100_d967.txt) shows every module class selected and served by cuda, `refused=-`; Transition
`n=118 served=cuda:1716` (before: 4 fp32 diffusion-head transitions on Triton). Step-graph fallbacks 0. Files: `data/f6_runs_h100_engine_d967.csv`,
`f6_summary_h100_engine_d967.csv`, `steps_h100_engine_d967.txt`. Run script of this session: `run_f6.sh` with `ARMS=miniworld`, `ENGINE_SRC=<overlay>`, `FFENV=<paper-306 env>`.

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldForge `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F6 (this folder):** `data/f6_runs_b200.csv` (24 runs), `data/f6_summary_b200.csv` (8 rows = AF3 / Protenix v1 x PyTorch, PyTorch+compile, cuEquivariance+compile, engine), `data/steps_b200.txt`. Every arm fits all 48 augmentations in one pass on the 178 GiB card (`aug_chunk 0`, no ladder), 3 repeats each, original H100 stand-in pools. Seconds per example (median of 3): AF3 engine 1.150 / cuEq+compile 1.837 / PyTorch+compile 2.005 / PyTorch 3.260; Protenix v1 engine 1.140 / cuEq+compile 1.667 / PyTorch+compile 1.815 / PyTorch 3.268. Engine arm = label A (engine main f06154bc7: the fp32 Transition chain `ieee_f32_sm100` is served by cuda, the 24 Transition Triton calls per micro step of the clamp-only tree are gone; the 9 LayerNorm w64 Triton calls per micro step remain, `triton_calls` column 12 counts the table of the run). The clamp-only engine rows are archived on the B200 (`invalid_archive/clamp_only_engine_before_holes_fix/`), not in the CSV.
Note: the `engine_version` column says 2.5.11 (package metadata of the shared env); `commit` / `foldforge_commit` of the BENCH line is the checkout of that launch (a lane re-deploys the repository between tasks; the training script and its inputs are identical across the commits of this branch).
