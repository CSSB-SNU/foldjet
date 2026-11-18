# 5I28 kit baseline, H100 (fixed 2026-11-18)

The kit side of the [5I28 benchmark](../guides/BENCHMARK-5I28.md), fixed as the baseline later FoldForge / engine runs are compared against.
The kits do not use the engine, so these values stay valid across engine releases. Data:
[data/5i28_kits_baseline_20261118.csv](data/5i28_kits_baseline_20261118.csv) (one row per kit and n: value, mode, status, timed forwards,
every forward in order, template evidence, source run, job, node). This file is not edited afterwards.

## Protocol

- As the [steady record](5i28_steady_20261115.md): per (kit, n) cell, forward 1 (cold) and forward 2 are dropped, then 5 timed forwards; the value
  is their median. n = 1, 2, 4, 8, 12, 16 chains of 5I28 (128 tokens each), real MSA (P0113359, 6789 rows) + 4 distinct real templates
  (RoseTTAFold3 1 per chain, ESMFold2 none), 4 trunk passes, 200 steps, 5 samples. A cell counts only with its template evidence in the log.
- **Mode: `fast`; a cell that runs out of memory under `fast` is re-run in the kit's `big` mode** in a fresh process (column `mode`). Protenix-v1
  family big is sized with `PROTENIX_V1_BIG_SIZE_N_TOKEN` = the cell's token count.
- NVIDIA H100 80GB HBM3, one GPU per job. Harness: `~/miniworld-engine-scratch/bench5i28_1118_steady/` (fb.sh = fast -> big fallback).

## Sources and nodes

- 73 cells measured 2026-11-18 (node01). Boltz-2 n = 1..12 (5 cells) are the 2026-11-15 steady values (node02, job 54120): that kit was not
  re-run for sizes that did not run out of memory.
- Chai-1 n = 16 has 4 timed forwards (one fold ~8 min).

## Cells without a value

| kit | n | why |
|---|---|---|
| esmfold2 | 16 | fast: out of memory; big: kit error (`CUDA error: an illegal memory access`) |
| esmfold2_fast | 16 | fast: out of memory; big: failed |
| intellifold2 (af3_jax kit + IntelliFold overlay) | 12, 16 | fast and big: out of memory (XLA triangle attention, no kit kernel for 8 heads x 64) |

`intellifold2_notxla` (`MODEL_OPT_LEVERS_OFF=TRIATT_XLA`, the kit's stock flash path for triangle attention) runs every n and is the row to
compare IntelliFold-v2 against.

## Cells under big mode

| kit | n | value (s) |
|---|---|---|
| boltz2 | 16 | 33.722 |
| opendde | 16 | 482.831 |
