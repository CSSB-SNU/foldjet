# 5I28, the Anthropic kits re-measured (2026-11-11)

The kit side of the [5I28 benchmark](../guides/BENCHMARK-5I28.md): ten kit runs (nine kits plus Protenix v2 / OpenDDE as separate kits)
at n = 1, 2, 4, 8, 12, 16 chains of azurin (128 n tokens), real MSA and 4 distinct real templates, one H100 80GB per job. The FoldForge side
of the same day is [5i28_foldforge_20261111.md](5i28_foldforge_20261111.md); this record is what its "Kits" table was taken from.
Data: [data/5i28_anthropic_kits_20261111.csv](data/5i28_anthropic_kits_20261111.csv) (every cell: warm min / median / max, status, job,
node, source log) and [data/5i28_anthropic_kits_20261111.json](data/5i28_anthropic_kits_20261111.json) (every forward of every cell, in
log order, first = cold). Harness: [`benchmarks/5i28_kits/`](../../benchmarks/5i28_kits/README.md). This file is not edited afterwards.

Written 2026-11-12 from the measurement directory `~/miniworld-engine-scratch/bench5i28_1111` (not in the repo; its `NOTES_authoritative.md`
names which files are authoritative). Everything below was read from that directory's done markers, kit logs and probe files.

## Conditions

- **Hardware:** `NVIDIA H100 80GB HBM3`, driver 575.51.03, one GPU per job; `node01` (AF3-JAX, Chai-1) and `node02` (all others).
  Slurm `-c 16 --hint=nomultithread` (16 physical cores); the kit process pinned with `taskset` to those 16 CPUs, `OMP/MKL/OPENBLAS/NUMEXPR`
  threads 16. Dedicated cores per job, but the node is shared:
  node CPULoad at job start 4-50 (node02) and 59-102 (node01) of 224 CPUs.
- **Container:** Ubuntu 22.04 Singularity image with the CUDA 13.0 forward-compat libcuda on the 575 driver (`cu130run.sh`), the kits' pinned
  stack (cu130 wheels). Kit **fast mode** everywhere (`ACTIVE mode=fast` in the Protenix-family logs).
- **Sampling:** 4 trunk passes, 200 steps, 5 samples, bf16 (kit defaults where a kit has no bf16 switch). Seeds: Protenix family 0-3
  (warm = seeds 1-3); AF3-JAX 6/6/6/5/4/4 seeds for n = 1/2/4/8/12/16; the other kits 3-4 forwards per cell (see "Warm forward counts").
- **Inputs:** the real MSA (kit a3m: 6790 rows; AF3-JAX, Chai-1, RF3, ESMFold2 log 6789) and 4 distinct templates `5SYD_A 6KOL_A 5YT7_A 6TFD_A`.
  Every cell's marker shows `real=4/4` (Protenix family: `Found 4 templates`, `real=4 dummy=0`); RoseTTAFold3 `1/1` (one template per chain);
  ESMFold2 takes none. Inputs are built by the scripts in `benchmarks/5i28_kits/` (`kits/*/mk*.py`, `make_inputs.sh`, `prep.py`, `in/`).
- **Kit versions** (installed packages in the kit venvs, read 2026-11-12): Protenix v1 `protenix 1.1.0` + `opt_core 0.5.228.0`, cuequivariance_torch
  0.11.1; Protenix v2 `protenix 2.0.0`; OpenDDE `opendde 1.1.1` + `opendde_opt 0.2.72`, cuequivariance_torch 0.10.0; ESMFold2 `esm 3.3.0` +
  `esmfold2_opt 0.3.27`, torch 2.13.0; AF3-JAX `alphafold3_open 3.1.4` + `af3_jax_opt 0.3.39`; AF3-Torch `alphafold3_open 3.1.4` +
  `af3_torch_opt 0.2.16`, torch 2.13.0; OpenFold3 `openfold3 0.5.0` + `openfold3_ob0_opt 0.8.18.1`, torch 2.10.0, cuequivariance_torch 0.10.0;
  Boltz-2 `boltz 2.2.1` + `boltz2_opt 0.3.36`, torch 2.12.0; Chai-1 `chai_lab 0.6.1` + `chai1_opt 0.4.33`, torch 2.13.0; RoseTTAFold3
  `rc_foundry 0.2.1.dev13+g4010e3e2e`, torch 2.13.0. The kit trees are not git checkouts (no commit ids); these are the only version records.
  Which venv each kit's run script activates was read from the scripts, not from the processes.
- **Timer span per kit** (the protocol's "forward-only" span, per the 2026-10-29 kit forward-definition audit): Protenix v1/v2, OpenDDE and
  OpenFold3 `PHASE total_s`; Boltz-2 `PHASE fwd_s` (`Boltz2.forward`, not `predict_step`); AF3-Torch `ITEM forward_s`; AF3-JAX "Running model
  inference with seed S took X seconds" (jitted model call plus the device-to-host copy); Chai-1 the forward-only span (the kit exits 1
  after its forwards); RF3 and ESMFold2 a `torch.cuda.synchronize()` timer around the model forward (`fwdtimer.py`, `esm_fwd_multi.py`).
- **Warm median:** the median of every forward after the first; an even count is the mean of the middle two.

## Method per kit (how a cell was run)

| kit | how | jobs |
|---|---|---|
| Protenix v1, Protenix v2, OpenDDE | **one process per n**, n ascending, inside one `queue3.sh` job (`queues/nmajor_*.txt`, markers `done_sep/`) | 46678, 46679, 46680 |
| Boltz-2, ESMFold2 | **one process, n ascending** (`run_multi.sh`, `esm_fwd_multi.py`); a cell that ran out of memory was retried in a fresh process | 46672, 46671 |
| AF3-JAX, AF3-Torch, OpenFold3, RF3, Chai-1 | one process per cell, one single-GPU job per kit (`queue3.sh`, markers `done_final/`) | 46657, 46658, 46659, 46661, 46656 |

Why the Protenix family is one process per n: the kit loops seeds outermost over a multi-item input, so n-major in one process interleaves
shapes (seed 0 of every n, then seed 1 of every n, ...) and its sampler graph cache is clamped to one entry. That inflated small n by about
15% (Protenix v1 n=1..4, run `done_seedmajor_multi/`, evidence only, not a 5I28 number). Around every cell the harness ran a host probe
(`probe.py`: a 2e7-iteration Python loop, a 4096^2 fp32 matmul, a 2 GiB copy) before and after, kept `scontrol show node` (CPUAlloc, CPULoad),
a GPU clock log every 5 s and a thread census every 60-120 s (`probe/` in the measurement directory; not copied here, 50 cells x 4 files).

## Kit warm median (s), with the 2026-10-29 kit value and the change

Cell = this run's median (2026-10-29 median, from the engine commit `71c800225`, `5i28_benchmark_all_models_20261029.md`) and the change.

| kit | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3_jax | 1.040 (1.040) +0% | 1.580 (1.570) +1% | 2.890 (2.900) -0% | 7.390 (7.465) -1% | 15.160 (15.120) +0% | 27.390 (27.360) +0% |
| af3_torch | 0.986 (1.044) -6% | 1.177 (1.304) -10% | 1.949 (1.952) -0% | 5.401 (5.400) +0% | 11.924 (11.753) +1% | 22.243 (22.131) +1% |
| openfold3 | 0.872 (0.877) -1% | 1.196 (1.351) **-11%** | 2.280 (2.364) -4% | 6.090 (6.098) -0% | 13.104 (13.096) +0% | 24.096 (24.172) -0% |
| rosettafold3 | 2.182 (2.275) -4% | 3.341 (3.397) -2% | 4.185 (4.537) -8% | 9.358 (9.122) +3% | 17.361 (17.301) +0% | 29.779 (29.636) +0% |
| chai1 | 4.556 (3.855) **+18%** | 4.165 (3.861) +8% | 10.093 (8.315) **+21%** | 27.501 (22.150) **+24%** | 51.804 (43.848) **+18%** | 89.661 (74.383) **+21%** |
| esmfold2 | 0.311 (0.308) +1% | 0.565 (0.564) +0% | 1.277 (1.503) **-15%** | 3.919 (3.878) +1% | 9.407 (9.069) +4% | oom (OOM) |
| boltz2 | 0.674 (0.694) -3% | 1.182 (1.183) -0% | 2.692 (2.691) +0% | 8.184 (8.152) +0% | 18.011 (17.906) +1% | oom (OOM) |
| protenix_v1 | 1.156 (1.138) +2% | 1.302 (1.315) -1% | 2.130 (2.191) -3% | 6.264 (6.262) +0% | 14.459 (14.476) -0% | 31.655 (31.658) -0% |
| protenix_v2 | 0.669 (0.666) +0% | 1.020 (1.015) +0% | 2.246 (2.242) +0% | 7.596 (7.519) +1% | 17.639 (17.719) -0% | 38.462 (41.716) -8% |
| opendde | 4.073 (4.321) -6% | 3.715 (3.770) -1% | 7.034 (7.077) -1% | 23.745 (23.840) -0% | 58.994 (59.233) -0% | oom (OOM) |

Cells that ran out of memory (`oom`): **ESMFold2 n=16**, **Boltz-2 n=16** (Boltz-2's `predict_step` swallowed the OOM; the cell has no forward) and
**OpenDDE n=16** (`CUDA out of memory. Tried to allocate 10.99 GiB`, 76.5 of 79.2 GiB in use; the marker says `exit=1`, the log says OOM).
The 2026-10-29 run had the same three cells out of memory.

## Kit / FoldForge (FoldForge side: [2026-11-11 record](5i28_foldforge_20261111.md); > 1: FoldForge faster)

| kit | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3_jax / af3 | 2.26 | 2.55 | 2.34 | 2.13 | 1.90 | 1.85 |
| af3_torch / af3 | 2.14 | 1.90 | 1.58 | 1.55 | 1.50 | 1.50 |
| openfold3 | 1.88 | 1.83 | 1.69 | 1.51 | 1.39 | 1.35 |
| protenix_v1 / protenix1 | 2.61 | 2.10 | 1.65 | 1.62 | 1.59 | 1.84 |
| protenix_v2 / protenix2 | 1.42 | 1.35 | 1.20 | 1.16 | 1.11 | 1.24 |
| opendde | 6.06 | 2.84 | 1.98 | - | - | - |
| boltz2 | 1.37 | 1.53 | 1.64 | 1.60 | 1.51 | - |
| rosettafold3 | 4.38 | 4.50 | 2.97 | 2.30 | 1.86 | 1.70 |
| esmfold2 | **0.57** | **0.70** | **0.77** | **0.90** | 1.01 | - |
| chai1 | - | - | - | - | - | - |

`-`: the kit ran out of memory, or FoldForge has no number (Chai-1 every n; OpenDDE n >= 8 not measured on FoldForge). Computed here from the
FoldForge medians of the FoldForge record and this record's kit medians.

## Cells that differ from 2026-10-29 by more than 10%

- **OpenFold3 n=2 (-11%), AF3-Torch n=2 (-10%, at the limit):** the 10-29 values were the noisy ones. The 10-29 logs have confidence-head
  spikes (`conf_s` 0.233 / 0.048 / 0.149 s in forwards of the same cell); the samplers are the same in both runs. Today's warm forwards of these
  cells: OpenFold3 n=2 1.180 / 1.212 s, AF3-Torch n=2 1.177 / 1.177 / 1.184 s.
- **ESMFold2 n=4 (-15%):** today 1.273 / 1.277 / 1.435 s against 1.503 on 10-29; the other five cells agree within 4%. The cause on 10-29 was not
  looked at (unverified).
- **Chai-1, every n (+8% to +24%):** slower today on `node01`, and the cells with the largest increase (n=8 +24%, n=16 +21%) are the ones with
  2-3 forwards. The host probe of the Chai-1 job (46656) ran slower than any other job's: the 2e7-iteration Python loop took 0.98-1.44 s
  against 0.85-1.27 s on the other jobs (AF3-JAX, also node01, reached 1.27), node01 CPULoad 59-100. This is a correlation
  with a slower host (Chai-1 is eager Python-heavy), not a proven cause (unverified); n=12 and n=16 have one warm forward each.
- **Protenix v2 n=16 (-8%):** under the limit, listed because it is the largest drop of the family (10-29: 41.716).
- **Protenix v1 in one process (the earlier method):** Protenix v1 in one multi-item process was +15% at n=1..4 from shape
  interleaving, which is why the family is now one process per n. The 10-29 Protenix cells were not interleaved and agree with today's
  within 3%.

## Warm forward counts and cold cost

The cell median is only as good as its warm forwards. Warm forwards per cell (the CSV's `n_warm`):

| kit | n=1 | n=2 | n=4 | n=8 | n=12 | n=16 |
|---|---|---|---|---|---|---|
| af3_jax | 5 | 5 | 5 | 4 | 3 | 3 |
| af3_torch | 3 | 3 | 3 | **2** | **2** | **2** |
| openfold3 | 3 | **2** | **2** | **2** | **2** | **2** |
| rosettafold3 | 3 | **2** | **2** | **2** | **2** | **2** |
| chai1 | 3 | **2** | **2** | **2** | **1** | **1** |
| esmfold2 | 3 | 3 | 3 | 3 | 3 | - |
| boltz2 | 3 | **2** | **2** | **2** | **1** | - |
| protenix_v1, protenix_v2, opendde | 3 | 3 | 3 | 3 | 3 | 3 (opendde -) |

- With two warm forwards the "median" is their mean: RF3 n=8 is 9.687 and 9.030 s (median 9.358; the 7% spread is the largest of any cell).
  A cell with one warm forward has no spread at all (Chai-1 n=12 / n=16, Boltz-2 n=12).
- **AF3-JAX's second forward is not warm** at n <= 4: forwards 2.470 / 2.070 / 3.890 s against 1.040 / 1.58 / 2.89 s afterwards (n = 1 / 2 / 4);
  the median is of the later forwards and is not affected.
- **Cold first forward** (CSV `cold_s`, first entry of `forwards_s`). At n=1: Protenix v2 2.6 s, Boltz-2 2.9 s, Protenix v1 3.0 s, ESMFold2 32.7 s,
  OpenDDE 27.7 s, OpenFold3 43.4 s, RF3 68.8 s, AF3-Torch 193.9 s, AF3-JAX 195.7 s, Chai-1 355.4 s. For Boltz-2 and ESMFold2 (one process) the
  later cells' "cold" forward is the first forward of a new shape. In the Protenix-v1 logs of the final cells the cuequivariance
  `stock_attention_preload` note says 2.0-3.3 s (x1 2.0, x2 2.1, x4 3.3, x8/x12/x16 2.0); values of 13.6, 48.9 and 74.6 s appear only in earlier,
  non-final logs of the same cell (`x1.node01`, `x1.preflight_node01`, `x1.node02_c120`, `x1.node02`). The cold cost is not a compile: the JIT
  cache directory is a symlink to the 10-29 cache and no file in it is new. The split of the other kits' cold forwards was not examined.


## Corrections to the FoldForge record (the same data, read again)

- **OpenDDE n=16** is an out-of-memory (log: `CUDA out of memory. Tried to allocate 10.99 GiB`); the FoldForge record's "failed" is the marker's
  `exit=1`. No number either way.
- **Boltz-2 medians:** the done markers hold every `PHASE` line twice (the kit prints it on stdout and stderr). The marker's median is taken
  from that doubled list with the first copy of the cold forward dropped, which leaves the second copy of the cold forward in the list.
  The raw per-forward values give 1.182 (marker 1.184), 2.692 (2.695), 8.184 (8.188) at n = 2, 4, 8; n = 1 (0.674) and n = 12 (18.011) are
  unchanged. This record's CSV is built from the de-duplicated lists; the FoldForge record keeps the marker values. Differences are <= 0.3%
  and move no ratio by more than 0.01.
- All other kit numbers in the FoldForge record equal this record's.

## What is not in this record

- **Superseded runs, evidence only:** four kits at once in one `-c 32` allocation on node01 (job 46644, `done_preflight_node01/`) read
  15-41% slower on the eager parts of the forward and the same on the sampler graph; it is not a 5I28 number and was replaced by the
  single-GPU jobs above. `x1.preflight_node01.log` was overwritten later (identical to `x1.node02_c120.log`) and is not authoritative.
  The seed-major multi-item Protenix run (`done_seedmajor_multi/`) is likewise evidence of the interleaving effect only.
- **Not measured:** the kits were not run at 10 trunk passes
  (see the 2026-10-29 Protenix record in the engine commit); no repeated job per cell (every cell is one job; the spread inside a cell is the only
  noise estimate).
