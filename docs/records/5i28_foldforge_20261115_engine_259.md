# 5I28, FoldForge on miniworld-engine 2.5.9, H100 (2026-11-15)

The FoldForge side of the [5I28 benchmark](../guides/BENCHMARK-5I28.md) re-measured on the released engine 2.5.9 (2.5.8 + the LayerNorm speed branch `ln-speed-1.05`
and the RTX 5090 gate fixes), all nine models. Data: [data/5i28_foldforge_20261115_engine_259.csv](data/5i28_foldforge_20261115_engine_259.csv) (every cell with its
warm forwards, the job, the 2.5.8 value and ratio, and the kit value with its ratio; the kit values are the ones carried in the
[2.5.8 record](5i28_foldforge_20261114_engine_258.md), the kits were not re-run). This file is not edited afterwards.

## Conditions

- **Hardware / software:** NVIDIA H100 80GB HBM3, node02 (every job), torch 2.13.0+cu129. `miniworld-engine` and `miniworld-engine-kernels` 2.5.9, loaded from a fresh
  FoldForge pixi environment (scratch worktree `~/foldforge-scratch/ff259`, `pixi install --frozen`; installed versions checked: engine 2.5.9, sm90 kernels 2.5.9).
  FoldForge `4f2f329` (clean tree, the 2.5.9 pin; the code is the 2.5.8 record's `e3b6eb3` plus the training-side commits, none of which the benchmark touches).
- **Protocol:** as in the [2.5.8 record](5i28_foldforge_20261114_engine_258.md) (real MSA P0113359, 4 distinct real templates, 4 trunk passes, 200 steps, 5 samples,
  bf16, `compile` + CUDA graphs, 3 warm repeats, one process per model, n ascending, per-size memory release). Number: `model_seconds_warm_median`.
- **Run directory:** `~/miniworld-engine-scratch/bench5i28_ff/run_1115_259`; compile caches `~/miniworld-engine-scratch/bench5i28_ff/cache_engine` (the warm cache of the
  2.5.8 run; the engine's kernels are prebuilt, the torch.compile cache is keyed by the code).
- **Concurrency (not the protocol's one-measurement-per-node rule):** three chains of jobs at the same time on node02, each job on its own GPU with 16 pinned physical
  cores: A = af3, openfold3, protenix1; B = protenix2, boltz2, rosettafold3; C = esmfold2, opendde, chai1. Other jobs (the engine release checks, other users) shared the node.
  Two training jobs ran on node01 at the same time, not on node02. The per-model `host.txt` / `gpu_clocks.csv` hold the node load and the SM clocks.
- **Differences against the 2.5.8 run:** engine 2.5.8 -> 2.5.9 only (same FoldForge code for the benchmark path).

## FoldForge, warm median (s); in brackets the 2026-11-14 value (engine 2.5.8)

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3 | 0.377 (0.381) | 0.576 (0.577) | 1.168 (1.180) | 3.372 (3.368) | 7.790 (7.781) | 14.388 (14.344) |
| openfold3 | 0.391 (0.392) | 0.605 (0.607) | 1.287 (1.292) | 3.919 (3.917) | 9.308 (9.306) | 17.783 (17.386) |
| protenix1 | 0.351 (0.351) | 0.554 (0.558) | 1.216 (1.210) | 3.729 (3.721) | 8.830 (8.897) | 16.607 (16.706) |
| protenix2 | 0.386 (0.389) | 0.690 (0.690) | 1.783 (1.787) | 6.278 (6.341) | 15.513 (15.562) | 31.145 (31.236) |
| boltz2 | 0.394 (0.394) | 0.669 (0.665) | 1.486 (1.488) | 4.755 (4.778) | 11.498 (11.513) | 22.052 (21.983) |
| rosettafold3 | 0.419 (0.421) | 0.637 (0.644) | 1.349 (1.357) | 3.913 (3.934) | 9.526 (9.598) | 17.505 (17.460) |
| esmfold2 | 0.245 (0.248) | 0.433 (0.434) | 1.019 (1.021) | 3.199 (3.195) | 7.202 (7.215) | 13.263 (13.753) |
| opendde | 0.594 (0.597) | 1.211 (1.229) | 3.503 (3.505) | 13.897 (13.119) | 32.697 (32.763) | 64.424 (63.781) |
| chai1 | 0.361 (0.364) | 0.713 (0.719) | 2.004 (2.001) | 7.438 (7.483) | 18.958 (19.605) | 35.982 (36.685) |

## Reading

- The numbers are unchanged from 2.5.8: the ratio 2.5.8 / 2.5.9 is 0.98 - 1.04 in every cell except OpenDDE n = 8 (0.94: 13.119 -> 13.897 s, while its n = 12 and n = 16
  are 1.00 and 0.99; a single-cell excursion, not repeated in the neighbouring sizes). The largest gains are ESMFold2 n = 16 (13.753 -> 13.263, 1.04) and Chai-1 n = 12
  (19.605 -> 18.958, 1.03).
- The 2.5.9 LayerNorm change is a kernel-level speed branch for the cases where the engine lost to the baselines in the module benchmarks; at model level it is
  below the run-to-run noise (concurrent jobs, three chains on the node).
- All 54 cells are `ok` (no out-of-memory, no retry).
