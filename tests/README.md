# Tests

Run `pytest tests` from the repository root. Tests are grouped by responsibility:

| Directory | Scope |
|---|---|
| `data/` | CCD, MSA/template LMDB, input/output contracts |
| `models/` | Loading, prediction, composition and frozen forward equivalence |
| `kernels/` | Backend dispatch, fused operators and precision contracts |
| `runtime/` | Bucketing, execution, cache supervision and output destinations |
| `utils/` | Shared numerical, serialization and process-state utilities |
| `diffusion/` | `foldjet.diffusion` stays self-contained; alignment defaults |
| `guidance/` | TFG guidance (Vina) |
| `train/` | the training recipe, step compile / graph, checkpointing, distributed bench |
| `references/` | frozen forward oracles and source identities used by regressions (snapshotted 2026-09-13) |
| `release_references/` | each family's structures from its own released code, the ground truth of `foldjet validate` ([README](release_references/README.md)) |

`support.py` owns repository and reference paths. `conftest.py` gives persistence
tests temporary run roots; tests never write into real prediction runs. GPU tests
skip when no GPU is allocated. Execute them on a Slurm compute node. Prepared
large input assets live under `$FOLDJET_HOME/benchmarks`, not inside the test package.
