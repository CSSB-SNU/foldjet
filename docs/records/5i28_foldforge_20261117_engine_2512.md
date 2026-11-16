# 5I28, FoldForge on miniworld-engine 2.5.12, H100 (2026-11-17)

The FoldForge side of the [5I28 benchmark](../guides/BENCHMARK-5I28.md) re-measured on the released engine 2.5.12 (2.5.9 + the TriAttn 16-channel heads and
TriMul training of 2.5.10, the OPM large-depth route of 2.5.11, the bidirectional TriAttn / TriMul of 2.5.12), all nine models. Data:
[data/5i28_foldforge_20261117_engine_2512.csv](data/5i28_foldforge_20261117_engine_2512.csv) (every cell with its warm forwards, the job, the 2.5.9 value and
ratio, and the kit value carried from the 2.5.8 record; the kits were not re-run). This file is not edited afterwards.

## Conditions

- **Hardware / software:** NVIDIA H100 80GB HBM3, node02 (every job), torch 2.13.0+cu129, `miniworld-engine` and `miniworld-engine-kernels` 2.5.12 from a fresh
  FoldForge pixi environment (worktree `~/foldforge-scratch/ff_2512`, `pixi install --frozen`). FoldForge `251bc02` (the 2.5.12 pin `cd53364` plus the
  cuBLAS-workspace fix of cached CUDA graphs; outputs of this benchmark unchanged by it).
- **Protocol:** as in the [2.5.8 record](5i28_foldforge_20261114_engine_258.md) (real MSA P0113359, 4 distinct real templates, 4 trunk passes, 200 steps,
  5 samples, bf16, `compile` + CUDA graphs, 3 warm repeats, one process per model, n ascending, per-size memory release). Number: `model_seconds_warm_median`.
- **Run directory:** `~/miniworld-engine-scratch/bench5i28_ff/run_1117_2512`; compile caches `~/miniworld-engine-scratch/bench5i28_ff/cache_engine`.
- **Concurrency (not the protocol's one-measurement-per-node rule):** the nine jobs were submitted at once on node02, four GPUs free: four ran at a time.
  A MiniWorld training (4 GPUs) shared node02; FoldBench runs and two fine-tunes ran on node01.

## FoldForge, warm median (s); in brackets the 2026-11-15 value (engine 2.5.9)

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3 | 0.380 (0.377) | 0.575 (0.576) | 1.181 (1.168) | 3.380 (3.372) | 7.838 (7.790) | 14.474 (14.388) |
| openfold3 | 0.392 (0.391) | 0.605 (0.605) | 1.297 (1.287) | 3.883 (3.919) | 9.215 (9.308) | 17.333 (17.783) |
| protenix1 | 0.350 (0.351) | 0.558 (0.554) | 1.226 (1.216) | 3.728 (3.729) | 8.858 (8.830) | 16.726 (16.607) |
| protenix2 | 0.389 (0.386) | 0.690 (0.690) | 1.789 (1.783) | 6.298 (6.278) | 15.555 (15.513) | 31.538 (31.145) |
| boltz2 | 0.398 (0.394) | 0.672 (0.669) | 1.497 (1.486) | 4.811 (4.755) | 11.669 (11.498) | 22.323 (22.052) |
| rosettafold3 | 0.420 (0.419) | 0.642 (0.637) | 1.350 (1.349) | 3.904 (3.913) | 9.515 (9.526) | 17.375 (17.505) |
| esmfold2 | 0.246 (0.245) | 0.432 (0.433) | 1.022 (1.019) | 3.202 (3.199) | 7.203 (7.202) | 13.444 (13.263) |
| opendde | 0.589 (0.594) | 1.227 (1.211) | 3.474 (3.503) | 13.002 (13.897) | 32.180 (32.697) | 63.868 (64.424) |
| chai1 | 0.365 (0.361) | 0.716 (0.713) | 2.009 (2.004) | 7.465 (7.438) | 19.112 (18.958) | 36.146 (35.982) |

## Reading

- Unchanged from 2.5.9: the ratio 2.5.9 / 2.5.12 is 0.985 - 1.069 over the 54 cells, median 0.997. The one cell above 1.03 is OpenDDE n = 8 (13.897 ->
  13.002 s), the cell that was a single-cell excursion in the 2.5.9 record (13.119 s on 2.5.8). The 2.5.10 - 2.5.12 changes target routes 5I28 barely
  uses (d_pair 64 x 4-head TriAttn of template stacks, the bidirectional modules, OPM at MSA depth > 4096).
