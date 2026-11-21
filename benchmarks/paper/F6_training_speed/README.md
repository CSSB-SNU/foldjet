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
