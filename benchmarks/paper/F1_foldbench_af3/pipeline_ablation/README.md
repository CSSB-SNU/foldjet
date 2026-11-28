# F1 addendum: pipeline ablation (2026-11-27/28)

Reviewer question: the 14.5 → 10.0 H100 GPU-h saving of FoldJet over the Anthropic AlphaFold3 kit on FoldBench (1,522 targets, 5 trunk
seeds × 5 samples, bf16) mixes kernels and pipeline. This folder separates them, and measures the two runner features that only act on
small targets (concurrency, seed grouping) per target size.

All runs: H100 80GB, AlphaFold3 weights, bf16, `--recycles 11`, the FoldBench runner (`benchmarks/foldbench/fb_multi.py`) at commit
671be49 + the `--order input` patch (a1f6cc9), engine 3.0.10, the shared pixi env. The FoldJet pipeline features the kit does not have:

| Feature | Runner flag (production value) | What it does |
|---|---|---|
| Small-target concurrency | `--small-concurrency 3` + CUDA MPS | up to 3 worker processes per GPU while targets ≤ 512 tokens run |
| Seed grouping | `--group 5 --group-max-tokens 512` | the diffusion of the 5 trunk seeds of a target ≤ 512 tokens in ONE sampler run (batch 25); trunks stay one by one |
| Size-bucket ordering | `--order total` | targets sorted by (token bucket, atom bucket), largest first; CUDA graphs kept per bucket |

Both sides keep feature prefetch and asynchronous CIF writing (the kit has them too), so they count on the engine side.

## 1. Cumulative ablation, full FoldBench (`data/ablation_gpuh.csv`)

Each run: all 1,522 targets, two shards on two GPUs (`run_ff.sbatch` LAUNCH=shard; the baseline LAUNCH=multi); GPU-h = sum of the shard walls.

| Run | Features on | GPU-h | Step | Contribution |
|---|---|---|---|---|
| Anthropic kit (F1) | — | 14.50 | | |
| ③ `af3_bf16_abl3_noconc_nogroup_nosort` | none (alphabetical order, as the kit) | **11.73** | 14.50 → 11.73 | engine (kernels, compile/graphs): **−2.77** (61 %) |
| ② `af3_bf16_abl2_noconc_nogroup` | ordering | **10.41** | −1.32 | size-bucket ordering: **−1.32** |
| ① `af3_bf16_abl1_noconc` | ordering + grouping | **10.37** | −0.04 | seed grouping: **−0.04** |
| baseline `af3_bf16_multi_3010` | all three | **9.97** | −0.40 | concurrency + MPS: **−0.40** |

Total saving 4.53 GPU-h = engine 2.77 (61 %) + pipeline 1.76 (39 %: ordering 1.32, concurrency 0.40, grouping 0.04).

- ③ lost 6 targets to CUDA out-of-memory (8oqr, 8oqo, 8tbs, 8xc1, 8xw8, 8z85: 2,194-2,408 tokens; all fine in the baseline): without
  ordering, the largest targets arrive between small ones, after graphs and caches of other buckets. They were run again with the same
  settings (job 67662, `..._retry`), all 6 passed; their per-target times (start → done, model load excluded, 2,006.5 s) are in ③.
  That retry shared node01 with other GPU work and ran 1.7× the cost estimate; with the estimate instead, ③ = 11.55 and ordering = 1.14.
- Node: the baseline ran on node02, ①②③ on node01.
- Seed grouping in the baseline: 796 of the 977 targets ≤ 512 tokens were grouped by 5, 4 by 2, 177 not at all: late in the run (≥ 04:07)
  three workers shared each GPU (MPS) and a batch-25 diffusion ran out of memory; the runner then halves the group cap of that size bucket
  (5 → 2 → 1, `groupcap` events). Concurrency and grouping compete for the same memory; concurrency wins (section 3).

## 2. Seed grouping vs size, 5I28 (`data/seed_group_5i28_h100.jsonl`, `seed_group_5i28.py`)

5I28 (n identical chains × 128 tokens, real MSA + 4 templates), FoldBench worker settings (bf16, 200 steps, compile + CUDA graphs +
bucketing). Per round: 5 trunks kept, then the diffusion one by one (5 × batch 5) vs grouped (1 × batch 25), order alternating;
7 rounds, first 2 dropped, median. Run with the ablation code (`import foldforge`; identical otherwise), jobs 67679-67681, node01.

| n | tokens | one by one | grouped | diffusion saving | diffusion share of GPU time in FoldBench* | target saving (est.) |
|---|---|---|---|---|---|---|
| 1 | 128 | 1.92 s | 1.38 s | **28 %** | 61 % | ~17 % |
| 2 | 256 | 2.64 s | 2.23 s | **16 %** | 55 % | ~9 % |
| 3 | 384 | 3.56 s | 3.33 s | 6 % | 48 % | ~3 % |
| 4 | 512 | 4.51 s | 4.33 s | 4 % | 45 % | ~2 % |
| 6 | 768 | 6.85 s | 7.19 s | **−5 %** | 35 % | ~−2 % |

\* from ② `run.json` stages (trunk vs diffusion seconds) by token range. n=8 (1,024) hit the job time limit. The trunk times in the jsonl
include CPU featurisation (the FoldBench worker prefetches it), so the target saving uses the FoldBench diffusion share instead.
The 512-token cap of `--group-max-tokens` is right: above it, grouping is slower.

## 3. Concurrency × seed grouping per size, warm (`data/concurrency_warm_*.csv`)

One H100 per run, the real runner on real FoldBench targets of ONE size bucket per range (`data/lists_warm/`): 3 warm-up targets first
(`--order input`; one per worker captures its CUDA graphs), then the measured targets. Per target time = (last done − 3rd done) / targets
done after the 3rd. 3 repetitions, median; spread within ±3 % except k3 at 257-384 (6.2-6.7 s). One job, a work queue over 3 GPUs, one
`srun` step (1 GPU, 12 cores, 40 GB) per run (`run_concurrency_warm.sbatch`, job 67779); `collect_concurrency_warm.sh`.

| Range (bucket, targets measured) | k1 (none) | k1g5 (grouping) | k3 (3 workers, MPS) | k3g5 (3 workers + grouping) |
|---|---|---|---|---|
| ≤ 256 (256/2048, 64) | 5.43 s | 4.77 s (1.14×) | 3.67 s (**1.48×**) | 3.58 s (**1.52×**) |
| 257-384 (384/4096, 36) | 8.36 s | 7.89 s (1.06×) | 6.24 s (**1.34×**) | 6.15 s (**1.36×**) |
| 385-512 (512/4096, 25) | 13.22 s | 12.73 s (1.04×) | 9.54 s (**1.39×**) | 9.83 s (1.35×) |
| 513-768 (640/8192, 17) | 18.13 s | 17.45 s (1.04×) | 14.24 s (**1.27×**) | 14.41 s (1.26×) |

- Concurrency (3 workers + MPS) is the main small-target optimisation: 1.27-1.48×. Seed grouping alone: 1.14× at ≤ 256 tokens,
  ≤ 1.06× above. Together they add only at ≤ 384 tokens; above, grouping adds nothing once the GPU is shared.
- k1 here agrees with the full runs (257-384: 8.36 s vs ~8.3 s per target in ②).
- Section 1 is consistent: targets ≤ 512 tokens are a small part of FoldBench GPU time, so concurrency saves 0.40 GPU-h (4 %) overall.

### Earlier, NOT warm (superseded; `data/concurrency_nowarm_*.csv`, `run_concurrency_queue.sbatch`)

Same idea with mixed buckets per range (`data/lists_nowarm/`) and no warm-up, 7 configurations (also k2, k4, k3 without MPS), 3
repetitions. Each new bucket captures CUDA graphs inside a ~5 min run, which inflates k1 (≤ 256: 367 s vs 317 s for the same targets in
the full run ②) and therefore every speed-up (grouping 1.33×, k3g5 1.64× at ≤ 256). Kept for the record; use section 3. What it does
show: 4 workers are no better than 3, and 3 workers without MPS (time slicing) lose more than half of the concurrency gain
(≤ 256: 1.32× vs 1.51×).

Two infrastructure notes from these runs:
- Several runs in one multi-GPU job must each see their GPU as device 0 (one `srun --exact --gres=gpu:1` step per run). With
  `CUDA_VISIBLE_DEVICES=2` for both the MPS daemon and its clients, every client failed "CUDA is unavailable" (job 67773, 28 runs).
- CPU actually used per run (sacct TotalCPU / elapsed): 1 worker 3.5-4.4 cores, 3-4 workers 9.4-11.2 cores; 12 cores per run suffice.

## Files

| File | Content |
|---|---|
| `data/ablation_gpuh.csv` | section 1 (shard walls, retry seconds, GPU-h) |
| `data/seed_group_5i28_h100.jsonl` | section 2, one line per size with all rounds |
| `seed_group_5i28.py` | section 2 script |
| `data/concurrency_warm_runs.csv`, `_median.csv`, `data/lists_warm/` | section 3 (per run / median / target lists: `.warm`, `.meas`, `.txt` = both in run order) |
| `run_concurrency_warm.sbatch`, `collect_concurrency_warm.sh` | section 3 job and collector |
| `data/concurrency_nowarm_*.csv`, `data/lists_nowarm/`, `run_concurrency.sbatch`, `run_concurrency_queue.sbatch`, `collect_concurrency_nowarm.sh` | superseded non-warm runs |

Run directories (not in the repository): `~/foldforge-scratch/paper/F1/h100/{af3_bf16_abl1_noconc, af3_bf16_abl2_noconc_nogroup,
af3_bf16_abl3_noconc_nogroup_nosort, af3_bf16_abl3_noconc_nogroup_nosort_retry, seed_group_5i28, concurrency, concurrency_warm}`; baseline
`~/foldforge-scratch/r3010/F1/h100/af3_bf16_multi_3010`.
