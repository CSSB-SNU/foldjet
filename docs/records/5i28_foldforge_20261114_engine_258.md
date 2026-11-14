# 5I28, FoldForge on miniworld-engine 2.5.8, H100 (2026-11-14)

The FoldForge side of the [5I28 benchmark](../guides/BENCHMARK-5I28.md) re-measured on the released engine 2.5.8, all nine models. Data:
[data/5i28_foldforge_20261114_engine_258.csv](data/5i28_foldforge_20261114_engine_258.csv) (every cell with its warm forwards, the job, and the kit value of
the [2026-11-11 kit record](5i28_anthropic_kits_20261111.md) with the ratio). This file is not edited afterwards.

## Conditions

- **Hardware / software:** NVIDIA H100 80GB HBM3, node02 (every job), torch 2.13.0+cu129. `miniworld-engine` and `miniworld-engine-kernels` 2.5.8, loaded from
  the FoldForge pixi environment's installed wheels (`meta.json`: `miniworld_engine_loaded_from`). FoldForge `e3b6eb3` (clean tree, the 2.5.8 pin).
- **Protocol:** `benchmarks/5i28_ff/` as in [the 2026-11-11 record](5i28_foldforge_20261111.md): real MSA (P0113359, 6790 rows), 4 distinct real templates
  (RoseTTAFold3 one per chain, ESMFold2 none), 4 trunk passes (AF3 `recycles` 3), 200 steps, 5 samples, bf16, `compile` + CUDA graphs (`scope: trunk`),
  3 warm repeats, one process per model, n ascending, per-size memory release. Number: `model_seconds_warm_median`.
- **Run directory:** `~/miniworld-engine-scratch/bench5i28_ff/run_1114_258` (cells, `host.txt`, `gpu_clocks.csv` per model); compile caches
  `~/miniworld-engine-scratch/bench5i28_ff/cache_engine` (warm, shared with the earlier runs).
- **Concurrency (not the protocol's one-measurement-per-node rule):** AF3 ran alone. The other eight models ran as three chains of jobs at the same time on
  node02, each job on its own GPU with 16 pinned physical cores: A = openfold3, protenix1, rosettafold3, esmfold2; B = protenix2, boltz2; C = opendde (n <= 12),
  chai1. OpenDDE n = 16 was run afterwards in its own job (52506) with no other job of this run alive. Other users' jobs shared the node's other GPUs. The
  timings of the concurrent models may be slightly pessimistic; the per-model `host.txt` / `gpu_clocks.csv` hold the node load and the SM clocks.
- **Code differences against the 2026-11-11 run:** engine 2.5.1 -> 2.5.8, FoldForge `6bbca72` + commits -> `e3b6eb3` (public capture API, exact OPM divisor,
  token-DiT conditioning hoist restored by 2.5.6). The change of the numbers is the sum of these, not the engine alone.

## FoldForge, warm median (s); in brackets the 2026-11-11 value (engine 2.5.1)

| model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3 | 0.381 (0.460) | 0.577 (0.619) | 1.180 (1.234) | 3.368 (3.474) | 7.781 (7.965) | 14.344 (14.788) |
| openfold3 | 0.392 (0.464) | 0.607 (0.653) | 1.292 (1.351) | 3.917 (4.027) | 9.306 (9.424) | 17.386 (17.800) |
| protenix1 | 0.351 (0.443) | 0.558 (0.620) | 1.210 (1.293) | 3.721 (3.877) | 8.897 (9.116) | 16.706 (17.202) |
| protenix2 | 0.389 (0.470) | 0.690 (0.754) | 1.787 (1.871) | 6.341 (6.543) | 15.562 (15.943) | 31.236 (31.101) |
| boltz2 | 0.394 (0.493) | 0.665 (0.774) | 1.488 (1.643) | 4.778 (5.105) | 11.513 (11.919) | 21.983 (22.732) |
| rosettafold3 | 0.421 (0.498) | 0.644 (0.742) | 1.357 (1.411) | 3.934 (4.064) | 9.598 (9.309) | 17.460 (17.514) |
| esmfold2 | 0.248 (0.542) | 0.434 (0.810) | 1.021 (1.663) | 3.195 (4.378) | 7.215 (9.331) | 13.753 (15.285) |
| opendde | 0.597 (0.672) | 1.229 (1.307) | 3.505 (3.558) | 13.119 (failed) | 32.763 (failed) | 63.781 (failed) |
| chai1 | 0.364 (not run) | 0.719 | 2.001 | 7.483 | 19.605 | 36.685 |

- The 2026-11-11 ESMFold2 row is before the FoldForge-side fixes of the [2026-11-12 record](5i28_esmfold2_20261112.md) (0.407 / 0.649 / 1.463 / 4.104 / 8.911 /
  14.816 s after them).
- **OpenDDE n = 16:** warm forwards 63.61 / 68.59 / 63.78 s (8 % spread); the job's log carries CUDA allocator out-of-memory warnings (12.9 GB request) on the way
  to a completed cell; the first attempt inside the chain was cancelled at the 10-minute limit while those retries ran. Read this cell with less trust than the others.

## Kit / FoldForge (> 1: FoldForge faster; kit values of 2026-11-11, not re-measured)

| kit / model | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3_jax / af3 | 2.73 | 2.74 | 2.45 | 2.19 | 1.95 | 1.91 |
| af3_torch / af3 | 2.59 | 2.04 | 1.65 | 1.60 | 1.53 | 1.55 |
| openfold3 | 2.23 | 1.97 | 1.76 | 1.55 | 1.41 | 1.39 |
| protenix_v1 / protenix1 | 3.29 | 2.33 | 1.76 | 1.68 | 1.63 | 1.89 |
| protenix_v2 / protenix2 | 1.72 | 1.48 | 1.26 | 1.20 | 1.13 | 1.23 |
| boltz2 | 1.71 | 1.78 | 1.81 | 1.71 | 1.56 | kit out of memory |
| rosettafold3 | 5.18 | 5.19 | 3.08 | 2.38 | 1.81 | 1.71 |
| esmfold2 | 1.25 | 1.30 | 1.25 | 1.23 | 1.30 | kit out of memory |
| opendde | 6.83 | 3.02 | 2.01 | 1.81 | 1.80 | kit failed |
| chai1 | 12.52 | 5.79 | 5.04 | 3.68 | 2.64 | 2.44 |

FoldForge is faster than the kit in every cell with a kit value; ESMFold2, slower than its kit on 2026-11-11 (0.57-0.90x at n <= 8), is now 1.23-1.30x.

## How far the kit numbers can be trusted (kit run-to-run variation)

The kit column is one set of measurements (2-5 warm forwards per cell, one node per kit, not the day of the FoldForge run). A separate analysis of the kits'
own variation (`~/miniworld-engine-scratch/kitvar-analysis/out/`, report summary below) found:

- **ESMFold2 n = 4 (1.503 s on 2026-10-29, 1.277 s on 2026-11-11):** measured cause: after a shape change the kit has one or two slow forwards (about 1.43-1.46 s
  against a steady 1.27 s); the 2026-10-29 harness ran one process per n (two slow forwards), the 2026-11-11 one an ascending process (one). Steady state over 5
  new processes: 1.266-1.273 s. A residual of about 5 % in the 2026-10-29 value is unexplained.
- **Chai-1 (+8 to +24 % on 2026-11-11):** node01 (host probe 1.3x slower, CPU load 59-100) against node02; Chai-1 n = 4 on node02, two jobs x 6 warm forwards: 7.69-7.84 s
  against 10.09 s (11-11, node01) and 8.315 s (10-29, node02). No kit could be run on node01 (all GPUs held by other users): the node effect rests on probes and cross-day numbers.
- **OpenFold3 n = 2 and AF3-Torch n = 2 (2026-10-29 higher by 11 % / 10 %):** the new repeats (OpenFold3 1.157-1.259 s, AF3-Torch 1.156-1.186 s) lie below the 10-29 values;
  the 10-29 runs shared a node with 2-3 concurrent processes without CPU pinning, and a four-kit concurrent run read 15-52 % slower (inference, not reproduced).
  About two thirds of the OpenFold3 gap was trunk and sampler, not the confidence head the 11-11 record named.
- **No change of kit or environment** was found between the dates for af3_torch, openfold3, esmfold2 and chai1.
- **Few warm forwards:** a cell with two warm forwards has a median equal to their mean; Chai-1 n = 12, 16 and Boltz-2 n = 12 have one.
- **Recommended kit protocol:** one standalone process per n, the first two forwards after the cold one dropped, at least six steady forwards, one named node with pinned
  CPUs and one kit per allocation, a host probe before and after, median of three processes. Measured spread with it: <= 1.7 % within a process, <= 4 % between processes.

So ratios close to 1 (protenix_v2 at n = 8-16, 1.13-1.23x) are inside the combined variation of the two sides; ratios of 1.5x and more are not.

## Not checked

Structures (these are timings); the same-day, same-node kit measurement; the A100 and the RTX 5090 (no 5I28 run on them with 2.5.8).
