# tools/

Developer and operator utilities: environment setup, quality checks, comparisons and audits, profiling and the
historical 4YX2 / 5I28 benchmark renderers. Nothing here is imported by `src/foldjet` (two tests load
`verify_prediction_artifacts.py` / mention `diff_dense_checkpoint.py` by path). Run them on allocated compute nodes,
never on a login node: `source tools/activate_env.sh` first, or submit the matching `.sbatch` wrapper.
Paper benchmark harnesses live in [`../benchmarks/`](../benchmarks/README.md); training drivers in [`../scripts/`](../scripts/README.md).

| file | group | what |
|---|---|---|
| `activate_env.sh` | environment | activate `.pixi/envs/default` and the C++ runtime |
| `setup_env.sbatch` | environment | `pixi install --locked` on a CPU allocation (FA2 with `FOLDFORGE_FLASH2=1`) |
| `check_quality.sh` | quality | Ruff lint / format and Pyright, as pre-commit runs them |
| `check_wheel.py` | quality | a built wheel's public imports, types and metadata (used by CI) |
| `typings/` | quality | Pyright stubs (`biotite-stubs`), `stubPath` in `pyproject.toml` |
| `references/` | validation | regenerate the release-reference structures in `tests/release_references/` ([its README](../tests/release_references/README.md)) |
| `fold_esmfold2.py` / `.sbatch` | inference | legacy `--target` entry point for `foldjet fold esmfold2` |
| `convert_dense_checkpoint.py` | checkpoints | checkout entry point of `foldjet weights convert` |
| `diff_dense_checkpoint.py` | checkpoints | diff a released parameter blob against a family's expected tree |
| `compare_dense_module.py` | checkpoints | run one dense-graph module on saved inputs against a reference dump |
| `check_live_inputs.py` | checks | is every input stream a model reads actually connected |
| `compare_backends.py` | checks | each backend against PyTorch for one model and input |
| `verify_prediction_artifacts.py` | checks | compare two prediction trees (confidence heads, CIF atom order) |
| `compare_execution_structures.py` | checks | graph / compile predictions against 0.25 A CA RMSD and 1 pLDDT point |
| `qualify_execution.py` | checks | graph replays against the same callable with capture off |
| `audit_model_boundaries.py` | audits | inventory model classes without importing Torch |
| `audit_af3_precision.py`, `audit_af3_batch.py` | audits | parameter / operand dtypes; batched vs single denoiser calls |
| `profile_af3_modules.py` / `.sbatch` | profiling | coarse per-module AF3 wall time |
| `sweep_diffusion_steps.py`, `evaluate_diffusion_sweep.py` | diffusion | sweep and score diffusion step counts |
| `benchmark_end_to_end.py` / `.sbatch` | benchmark (2026-09) | five execution modes per model |
| `merge_benchmark_rows.py` | benchmark (2026-09) | merge fill-in run roots into one |
| `render_benchmark_results.py` | benchmark (2026-09) | render the 4YX2 matrix (now in `docs/archive/`) |
| `render_length_benchmark.py` | benchmark (2026-09) | render `docs/benchmark_results.md` (replicated 5I28 + length ladder) |
| `check_structures.py`, `compare_af3_precision.py`, `collect_structure_issues.py` | benchmark (2026-09) | per-sample geometry / drift / experimental checks the renderers publish |

Removed in this tidy-up: `esmfold2_profile.sbatch` and `op_dispatch_audit.sbatch`, wrappers whose Python scripts were
deleted in commit `2df3eb5` (the sequence-layout removal). `audit_sample_axes.py` (described under Audits below) was deleted
in the same commit; its entry is kept for the archived benchmark records that cite it.

## Inference

The standard entry point is `foldjet fold <model> --spec <MiniWorld YAML>`,
optionally with `--config <runtime YAML>`. All predictors read `ccd_db` from the
input YAML and share `$FOLDFORGE_HOME/ccd/preprocessed_CCD.lmdb`; prepare it with
`foldjet ccd prepare` and check it with `foldjet ccd verify`. See
[the format contract](../docs/guides/MINIWORLD-FORMAT.md) and
[model integration](../docs/guides/MODEL-INTEGRATION.md) for input schemas,
checkpoint paths and complete commands.

Inference and benchmark commands accept `--trunk-seed` and `--diffusion-seed`
(defaults 0/0). Input/conformer/MSA/trunk randomness and diffusion
noise/augmentation use independent streams. Result JSON records both seeds and
`seed_policy: split-v1`; historical coupled-seed results are not rewritten. Use
a new run directory when changing seed policy.

Compatibility tools for reproducing the recorded validation runs:

- `fold_esmfold2.py` / `.sbatch`: same implementation as `foldjet fold esmfold2`
  through the legacy `--target` inputs. Live ESMC is the default; `--lm-source cache`
  reuses existing embeddings. Accepts `--ccd-db`.
- `verify_prediction_artifacts.py`: compare two prediction trees produced with
  identical inputs, including confidence heads and CIF atom order.
- `compare_execution_structures.py`: matched graph/compile predictions against
  0.25 A CA RMSD and one pLDDT point limits.
- `qualify_execution.py`: graph replays against the same callable with capture off.

## Model boundaries

- `audit_model_boundaries.py --out docs/archive/model-boundaries-<date>.csv`:
  inventory model classes with forward methods without loading the GPU stack.

## Profiling

- `profile_af3_modules.py` / `.sbatch`: AF3 stage timings for one complete forward
  on the qualification target. Eager execution with a CUDA synchronize around
  every hooked stage (conditioning encoder, trunk and its MSA/Pairformer stacks,
  diffusion head and its encoder/transformer/decoder, sampling loop, confidence
  and distogram heads). Writes `profile.json` and `profile.md` under the run
  directory. Inclusive stage wall time, not a kernel profile; the synchronizes
  make it an upper bound on the same stages inside the compiled/graph benchmark.

```bash
sbatch tools/profile_af3_modules.sbatch --out $FOLDFORGE_HOME/runs/af3-profile --repeats 2
```

## Benchmark

`benchmark_end_to_end.py --model <model>` defaults to five explicit modes for
AF3, OpenDDE, ESMFold2 and Protenix v2:

| Mode | Precision | Compile | Manual CUDA graph |
|---|---|---|---|
| `pytorch_eager_reference` | Model-default precision | no | no |
| `pytorch_compile_reference` | Model-default precision | yes | no |
| `pytorch_compile_bf16` | native BF16, FP32 norms | yes | no |
| `cuequiv_compile` | native BF16, FP32 norms | yes | no |
| `miniworld_graph` | native BF16, FP32 norms | yes | yes |

AF3 reference precision is `precision: af3_default`: input atom encoder and
diffusion FP32, trunk and confidence Pairformers BF16, norms and final confidence/
distogram projections FP32. Released FP32 weights are never rounded through BF16.
`recycles` means total trunk passes for every model: 10 gives 10 passes.
Historical AF3 measurements used the former additional-pass convention
(setting 10 ran 11); use 11 to reproduce that workload. Historical records and
their renderers retain the original measurement contracts.
The benchmark retains token buckets in multiples of 128, disabled templates,
five batched diffusion samples, and denoiser-only compile/graph scope. This is
a FoldJet backend comparison, not a measurement of the official JAX runtime.

For the other models, `precision: model_default` keeps FP32 parameter storage.
OpenDDE executes in FP32 without autocast. Protenix v2 uses BF16 autocast outside
its FP32 diffusion sampler. ESMFold2 uses BF16 autocast in the input/trunk,
confidence folding trunk and diffusion pair transitions; remaining diffusion
projections stay FP32. Their reference TF32 setting is enabled. These are the
released precision scopes reproduced in FoldJet, not the original applications.
All three native BF16 comparison modes use BF16 parameters except FP32 norms,
with autocast disabled.

`--modes` selects a subset. Explicit `pytorch_eager_fp32` and
`pytorch_compile_fp32` retain whole-model FP32 diagnostics. The historical
`pytorch_eager` and `pytorch_compile` aliases remain BF16-only.
`--backends` retains the legacy compile-and-graph comparison and cannot be
combined with `--modes`. Use `--variant protenix-v2` for the v2 checkpoint.

`render_benchmark_results.py --results RUN --docs docs [--audits-recorded DATE]
[--notes docs/benchmark_notes.md]` validates all 20 cases
and renders the five-mode latency table and bar chart. It rejects inconsistent
inputs, precision, sample/recycle counts and execution flags, and refuses to mix
legacy and split-seed measurements in one comparison. Timing definitions and
current results are in [benchmark results](../docs/benchmark_results.md).

`check_structures.py --results RUN --output RUN/structure_checks.json` writes the
per-sample geometry and same-index drift records for ESMFold2, Protenix v2 and
OpenDDE that the renderer publishes; `compare_af3_precision.py` does the same
for AF3 with an experimental comparison. Place both outputs, `precision_audits.json`
and `sample_axes_audits.json` in RUN before rendering.

`collect_structure_issues.py --results RUN --output DIR` copies only the samples
with a peptide C-N outside 1.0-1.7 A, a heavy-atom pair below 1 A, or CA drift
above 1.5 A from the same-index compiled reference. Each `DIR/<model>/sample-<i>/`
holds `ref.cif` beside `mini.cif`, `pytorch_bf16.cif`, `cueq.cif` or
`eager_ref.cif` and an `issues.md` naming the residues and atom pairs;
`DIR/README.md` is the combined index. Report CIF paths from
another host resolve to copies under `RUN/e2e/`.

## Audits

Run each inside an allocated GPU job. Their timings are diagnostic and must not
enter latency comparisons.

- `audit_af3_precision.py REPORT --spec INPUT --config CONFIG --out RUN`: records
  every parameter, observed Linear/GEMM operand dtypes, DiT module outputs and
  actual CUDA autocast state. Add `--model esmfold2|opendde|protenix` for the
  other adapters. Supply an eager configuration with `precision: bf16`,
  `af3_default` (AF3) or `model_default` (other models), one recycle, two
  diffusion steps and one sample.
- `audit_af3_batch.py REPORT --spec INPUT --config CONFIG --out RUN`: compares
  identical coordinates in one batched denoiser call against five independent
  calls, checks one call per step, and observes MiniWorld Q/K/V `num_aug=5` with
  shared pair bias. Use five samples, one recycle and two diffusion steps. The
  full benchmark also rejects sequential AF3 execution through observed wrapper
  call counts. Denoiser batches change random-number consumption relative to the
  old sequential sampler; same-seed old/new trajectories need not match.
- (removed in `2df3eb5`; kept for the archived records) `audit_sample_axes.py REPORT MODEL --spec INPUT --config CONFIG --out RUN`:
  ESMFold2, Protenix v2 and OpenDDE token kernel Q/bias shapes, excluding trunk
  triangle attention, and ESMFold2 SWA construction with `num_aug=5`. Use eager
  MiniWorld, five samples, one recycle and two steps; supply `--lm-cache` for
  ESMFold2. SWA's FlashAttention boundary flattens augmentation and structure
  batch; token attention retains `[A, B]`. Rectangular flat-atom windows use the
  shared batched PyTorch path because the engine pair-bias core currently
  supports square attention.
- `compare_af3_precision.py --results RUN --experimental 4yx2.cif --output
  quality.json --report af3_precision.md`: compares all samples with the compiled
  AF3-default reference (`pytorch_compile_reference`) and the experimental
  complex, and writes a latency chart and quality table. Feed a merged
  `e2e-af3.json` containing the five measured modes. Atom correspondence uses
  chain, label residue index, residue identity and atom name; no
  matching-by-coordinate or best-sample substitution is performed.
