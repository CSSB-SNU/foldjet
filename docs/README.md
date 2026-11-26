# Documentation

Start with the [root README](../README.md) (install, predictors, quick start). Every file under `docs/` is listed below.
Pages marked *dated* record a measurement or decision of their date and are not re-checked against the current code.

## Using FoldJet

| page | what |
|---|---|
| [guides/WEIGHTS.md](guides/WEIGHTS.md) | downloading, converting and verifying weights for every model; language models; offline import (in Korean) |
| [guides/CHECKPOINT-FORMAT.md](guides/CHECKPOINT-FORMAT.md) | the miniworld checkpoint format (fp32 engine-key `state_dict`), where the default file is found, the one-time `foldjet weights convert` |
| [guides/CHECKPOINTS.md](guides/CHECKPOINTS.md) | from a release file to a blob pinned in `checkpoints.lock`; which conversions reproduce the lock bit for bit |
| [guides/CHECKPOINT-VARIANTS.md](guides/CHECKPOINT-VARIANTS.md) | checkpoint variants per family |
| [guides/PYTHON-API.md](guides/PYTHON-API.md) | `load_model()` / `load_inputs()`, reusable models, trunk / diffusion test-time scaling, results, the CLI over the same API |
| [guides/CONFIG.md](guides/CONFIG.md) | every `Config` key (`mode`, `precision`, `engine.*`, compile / graph), environment variables that became config |
| [guides/MINIWORLD-FORMAT.md](guides/MINIWORLD-FORMAT.md) | input YAML, CCD / MSA-LMDB / template-LMDB formats, the shared MSA policy |
| [examples/inference/](examples/inference/) | example input (`1ubq.yaml`, `1ubq.fasta`) and runtime configs (`native-bf16`, `graph-bf16`, `compile-graph-buckets-bf16`, `operational-1ubq`) |
| [guides/BACKENDS.md](guides/BACKENDS.md) | `pytorch / cuda / triton / anthropic / cueq` and the `miniworld` default, `fallback / error`, precision `master / fp32`, comparing backends |
| [guides/EXECUTION-SCOPES.md](guides/EXECUTION-SCOPES.md) | what `compile` and `cuda_graph` wrap (`denoiser`, `trunk`) and the rules for captured code |
| [guides/TRAINING.md](guides/TRAINING.md) | Protenix v1 fine-tuning on the FoldJet graph: recipe, batch-source plug-in, what is implemented (in Korean); driver in [`scripts/`](../scripts/README.md) |

## Accuracy and validation

| page | what |
|---|---|
| [validation/README.md](validation/README.md) | how FoldJet is judged against each release's own code (`foldjet validate`), the rule per check, defects found |
| [validation/fast.md](validation/fast.md), [validation/exact.md](validation/exact.md) (+ `.json`) | the current verdicts for the two modes |
| [guides/MODEL-INTEGRATION.md](guides/MODEL-INTEGRATION.md) | per-family checkpoint evidence, release environments and validation scope (long; partly *dated*) |
| [findings/reference-conformer-pose-sensitivity.md](findings/reference-conformer-pose-sensitivity.md) | reference conformer and pose leak information into the trunk (explains the one failing OpenDDE row) |
| [bug-reports/](bug-reports/README.md) | upstream defects found while validating (Chai-lab RDKit 2025, ESMFold2 ESM-C weights, OpenDDE ligands, af3-any-model nucleic order); none filed yet |

## Benchmarks

| page | what |
|---|---|
| [guides/BENCHMARK-5I28.md](guides/BENCHMARK-5I28.md) | the 5I28 speed protocol (FoldJet vs the Anthropic kits, n = 1..16 azurin chains, real MSA + 4 templates, warm median) |
| [records/](records/) *dated* | 5I28 and FoldBench records, 2026-11-11 .. 11-18 (raw CSV / JSON in `records/data/`); files keep their pre-rename `5i28_foldforge_*` names: [kits 11-11](records/5i28_anthropic_kits_20261111.md), [FoldJet 11-11](records/5i28_foldforge_20261111.md), [A100 11-12](records/5i28_foldforge_a100_20261112.md), [atom-DiT bug 11-12](records/5i28_atom_blocks_check_20261112.md), [ESMFold2 11-12](records/5i28_esmfold2_20261112.md), [engine 2.5.8](records/5i28_foldforge_20261114_engine_258.md), [2.5.9](records/5i28_foldforge_20261115_engine_259.md), [steady protocol 11-15](records/5i28_steady_20261115.md), [2.5.12](records/5i28_foldforge_20261117_engine_2512.md), [FoldBench e2e 11-17](records/foldbench_e2e_20261117.md), [kit baseline 11-18](records/5i28_kits_baseline_20261118.md) |
| [benchmark_results.md](benchmark_results.md) *dated* | 2026-09 benchmark: 5I28 replicated on A100 / A6000, five execution modes at 512 tokens, length ladder to OOM; rendered by `tools/render_length_benchmark.py`, which inserts [benchmark_notes.md](benchmark_notes.md); figures and JSON in [assets/](assets/) |
| [../benchmarks/](../benchmarks/README.md) | the harnesses: paper F1 - F11, 5I28 (FoldJet and kit sides), FoldBench end to end |

## Developing FoldJet

| page | what |
|---|---|
| [guides/CODE-STRUCTURE.md](guides/CODE-STRUCTURE.md) | the directory guide: who owns what (FoldJet vs foldjet-engine), where a change goes |
| [guides/DATA-AND-EVALUATION.md](guides/DATA-AND-EVALUATION.md) | which module owns constants, CCD, featurization, inputs, judging, confidence decoding |
| [guides/CODE-QUALITY.md](guides/CODE-QUALITY.md) | Ruff / Pyright scope, numerical-code exceptions, compute-node regression checks |
| [ENGINE-UNSUPPORTED.md](ENGINE-UNSUPPORTED.md) *dated* | what the engine (2.5.x) did not support and what FoldJet does meanwhile; cited by source comments, keep the path |
| [ENGINE-REQUESTS.md](ENGINE-REQUESTS.md) *dated* | numbered requests to the engine team (2.5.0 - 2.5.6) |
| [assets/model_code_branches.dot](assets/model_code_branches.dot) | Graphviz diagram of the per-model code branches (2026-11-10, labels in Korean) |
| [../tools/README.md](../tools/README.md), [../tests/README.md](../tests/README.md) | developer utilities; test layout |

## History

[archive/](archive/README.md): dated integration, audit and benchmark records from 2026-09 to 2026-10, the engine 2.4.x
migration records and the pre-2.5.0 backend measurements, each with its raw JSON / CSV evidence. Not current.

The recorded 2026-09 benchmarks (before the team-gm dependency was removed, commit `f3e6128`) pinned `libs/team-gm` to `92a50c9`
(branch `foldjet/msa-bucket-1024`: the validated `a823f69` runtime plus a 1024-row MSA inference bucket); the 2026-09-16 release
used `a823f69` itself. To reproduce them, check out `f3e6128` and run `git submodule update --init --recursive`
(`--remote` would select newer, unmeasured runtime changes).
