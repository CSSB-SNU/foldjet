# FoldBench end to end: FoldForge (AF3, engine 2.5.12) vs af3kit, H100 node01, 2026-11-17

Harness of later runs: [benchmarks/foldbench](../../benchmarks/foldbench/README.md). This file is not edited afterwards.

| | af3kit (job 56150, 11-16 17:43) | FoldForge (jobs 56677 / 56678, 11-17 08:05) |
|---|---|---|
| shard s0 wall | 6.36 h | **5.27 h** |
| shard s1 wall | 6.27 h | **5.31 h** |
| finished targets | 754 + 752 = 1,506 (16 failed: OOM on the largest) | **761 + 761 = 1,522 (0 failed)** |
| end-to-end (slower shard) | 6.36 h | 5.31 h -> **1.20x** |

Setup both: FoldBench 1,522 targets, kit shards2 split (s0/s1), one process per GPU, DeepMind AF3 weights, 5 trunk_seeds x 5 samples, 200 steps,
11 trunk passes (kit: 10 recycles), MSA 16,384 rows, 4 templates, node01 with the two fine-tunes running. Kit: mode fast, alphabetical order.
FoldForge: FoldForge 399d0b4 (2.5.12 pin + cuBLAS-workspace fix 251bc02), the one-process-per-shard runner `fb_run_group.py` (5 trunk_seeds grouped
up to 1,024 residues; not in the repository, superseded by `benchmarks/foldbench/fb_multi.py`), bf16 master, compile + CUDA graphs (scope trunk),
supervisor restarts. Run directories (not in the repository): `ff2512/runs/{s0,s1}`; earlier runs kept: `runs_prefix_251bc02` (the cuBLAS-workspace
bug), `runs_chunked4h_251bc02` (4 h chunks, invalid for walltime).

Specs: `specs_all` (converted 2026-10-29, `n_trunk_samples: 1` / `template_dedup: false` since 2026-11-11). The template LMDB of
BioMolDB_20261014 is dated 2026-10-29 17:25, after that conversion: a rebuild with `benchmarks/foldbench/build_specs.py` (2026-11-17) gives the same
1,522 specs byte for byte except 24 targets that gain template records (keys the older database lacked): 7fzi 7oyp 7ucx 7yex 8ahn 8b9e 8ci9 8ekv 8hwi 8okj 8oqr 8orc 8out 8pu1 8pyi 8qoh 8r5t 8rue 8sao
8u8k 8vfe 8w8h 8wbt 8z2m (all `-assembly1`).

Handicaps of this FoldForge run (fixed in `fb_multi.py` / later commits, not in this number): both shards hit an OOM with CUDA graphs
(s0 10:16, s1 10:41) and ran the remaining ~90 / ~64 largest targets without graphs; grouping up to 1,024 residues (measured loss above
~512 tokens); size_proxy order (not bucket-sorted); MSAPairWeightedAveraging past 2,048 tokens on pytorch (fixed: 4c0aa72); plain fp32
LayerNorms (engine LayerNorm: ba8b27b); expandable_segments off (FoldForge default since the commit that added this record).
