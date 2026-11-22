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
