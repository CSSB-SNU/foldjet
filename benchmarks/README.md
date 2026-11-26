# benchmarks/

Measurement harnesses and their recorded data. Each folder has its own README with conditions and exact commands; outputs of new
runs go to scratch, only final CSV / record files are committed. Rules for every speed number: [the root README](../README.md#conventions).

| folder | what | record / results |
|---|---|---|
| [`paper/`](paper/README.md) | the paper experiments F1 - F11 (FoldBench GPU-hours and accuracy, 5I28 times, sample comparison, inference / training limits, training speed, fine-tune loss, FP8 accuracy), one folder per experiment, shared helpers in `common/` and `train_common/` | `paper/F*/data/` |
| [`5i28_ff/`](5i28_ff/README.md) | the FoldJet side of the 5I28 protocol ([BENCHMARK-5I28.md](../docs/guides/BENCHMARK-5I28.md)) | [`docs/records/`](../docs/records/) |
| [`5i28_kits/`](5i28_kits/README.md) | the Anthropic-kit side of the same protocol | [`docs/records/5i28_anthropic_kits_20261111.md`](../docs/records/5i28_anthropic_kits_20261111.md) |
| [`5i28_data/`](5i28_data/README.md) | the 5I28 inputs (MSA, templates) for both sides, about 2 MB | |
| [`foldbench/`](foldbench/README.md) | FoldBench end to end (1,522 targets, one job on 1..N GPUs, bucket-affine scheduler) | [`docs/records/foldbench_e2e_20261117.md`](../docs/records/foldbench_e2e_20261117.md) |

The 2026-09 4YX2 / replicated-5I28 benchmark (five execution modes) is run with `tools/benchmark_end_to_end.py` and rendered into
[`docs/benchmark_results.md`](../docs/benchmark_results.md); see [`tools/README.md`](../tools/README.md#benchmark).
