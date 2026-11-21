# F10: training limit (Protenix v1, MSA depth 1,024)

> **Removed from the paper (user decision, 2026-11-21).** The code and the H100 engine 3.0.6 results (`data/engine306/`) are kept as a record; no further runs.

Paper: new (limit of what one card can train). Arms: engine, PyTorch (+ `torch.compile`), cuEquivariance + `torch.compile`. Cards: H100, A100, RTX 5090.

## Questions

- **(a) longest crop.** With activation checkpointing (policy `all`) and the diffusion-augmentation chunk set as needed, the longest crop (tokens and atoms) that trains.
  Atoms follow the recipe ratio 4,096 atoms per 384 tokens (= 32/3 per token, rounded up to the 32-atom window): 768 tokens = 8,192 atoms.
- **(b) most Pairformer blocks.** At 384 tokens / 4,096 atoms and at 768 tokens / 8,192 atoms, **without** checkpointing, the largest number of trunk Pairformer blocks
  (native 48) that trains.

"Trains" = 3 optimizer steps of the Protenix recipe complete with finite loss and gradient norm (step graph, 4 recycles = the worst case for memory, mini-rollout and
confidence head included, fp32 diffusion head and fp32 pair-bias logits as always). Peak CUDA allocated / reserved memory is recorded at every point.

## Protocol

One **trial is one process** (`trial.sh`, via `../train_common/trial_main.py`, which runs `scripts/train_protenix1.py` unchanged; `PF_BLOCKS` cuts / extends the trunk
Pairformer stack, extra blocks are copies of the native ones: a memory / speed probe, not a trained model). An out-of-memory trial is a **recorded result**: it is never
retried with another shape. The search (`bisect.py`) per arm and part:

1. Walk up a fixed ladder (length: 384, 512, 640, 768, 1024, ... 8192; blocks: 48, 64, 96, 100 -- capped at 100, reaching it reads ">= 100") with the smallest-memory augmentation chunk (1) until the first
   failure (first rung failing: the search runs down from it), then **bisect** between the last success and the first failure (length on a 64-token grid, blocks to 1).
2. At the maximum found, the chunk ladder `0,24,12,6,3,1` (0 = all 48 augmentations in one pass) is run from large to small; the first chunk that trains is the setting
   "as needed", with its peak memory and step time.

A non-OOM failure (any error that is not out of memory) stops that search and is recorded as `error` (not counted as a limit).
Stand-in data: the pool of F6 (8 FoldBench targets, `pool_protenix1.pt`) padded to the trial's shape (TOKENS / ATOMS, MSA pool 16,384, 4 templates), MSA depth 1,024 rows
per recycle (`--msa-rows 1024`). The padded shape sets the memory; the trial does not need a real target of that length.

## Files

`trial.sh` (one trial), `bisect.py` (search driver, standard library only, resumable: `results.jsonl` is read back), `run_f10.sh` (all parts x arms on one card),
`h100.sbatch`, `a100.sbatch`, `rtx5090.sh`, `collect.py`.

## Output

`~/foldforge-scratch/paper/F10_training_limit/<card>/`: `results.jsonl` (every trial: arm, part, tokens, atoms, chunk, ckpt, blocks, status, peaks, step time, FoldForge commit,
engine version / file, GPU, host, Slurm job, log), `summary_<part>_<arm>.json`, `logs/`, `meta_<job>.txt`. `collect.py` writes `f10_trials_<card>.csv` (every trial) and
`f10_limits_<card>.csv` (per part x arm: the maximum, the first failing value, the chunk used, peak memory at the maximum).

## Commands (full campaign; started by the coordinator after the release and the FoldForge pin)

```
mkdir -p ~/foldforge-scratch/paper/F10_training_limit/h100/pool && cp ~/foldforge-scratch/paper/F6_training_speed/h100/pool/pool_protenix1.pt ~/foldforge-scratch/paper/F10_training_limit/h100/pool/
sbatch benchmarks/paper/F10_training_limit/h100.sbatch          # ssh cssb3: a100.sbatch;  ssh cssb-rtx: ~/mwe/gpu.sh is taken per trial by rtx5090.sh
python3 benchmarks/paper/F10_training_limit/collect.py --out-dir ~/foldforge-scratch/paper/F10_training_limit/<card>
```

Smoke (one trial per card, plumbing only): `SMOKE=1 ENGINE_SRC=<engine source> sbatch --time=00:10:00 --export=ALL h100.sbatch`. The job time of the full campaign is long
(about 3 arms x 3 parts x ~12-20 trials); `results.jsonl` makes it resumable, so a deadline is continued by submitting the same job again.

## Status (engine 3.0.6 campaign, 2026-11-21)

- **H100** (`data/engine306/f10_limits_h100.csv`, `f10_trials_h100.csv`; Slurm 62836 / 62837 / 62838; block ladder capped at 100 by the user's rule).
- **Not engine-CUDA-only (rows `engine_cuda_only = no`; not re-measured). Engine defect:** in every engine-arm trial the fp32 `Transition` calls are refused by the engine's H100 cuda backend ("float32 activations run on
  TF32 tensor cores only (torch does not allow TF32 here)") and run its triton backend (588 of 2,280 Transition calls in the first micro step at 384 tokens).
  The engine arm is therefore not all-cuda on the H100; recorded, to be fixed in the engine (the 3.0.4 IEEE fp32 chain covers the sm_80 family only).
