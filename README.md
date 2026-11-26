# FoldJet

Renamed from FoldForge (`import foldforge` and the `foldforge` command still work; `import foldjet` is the name now). Runs from before the rename keep their recorded `foldforge_commit` columns and the home directory stays `~/.cache/foldforge` (`$FOLDFORGE_HOME`).

**Many structure predictors, one environment, one set of kernels.**

FoldJet runs eleven AlphaFold3-family structure predictors through one PyTorch graph on one GPU engine,
foldjet-engine (fused kernels and the modules that wrap them). Every
predictor reads the same MiniWorld-format inputs and one CCD database; a predictor is a `DenseSpec` row stating where its
release differs from AF3 (equations, checkpoint layout, confidence head), not a second implementation. The repository also
holds the benchmark harness of the paper (`benchmarks/paper/F1`..`F11`).

Inference uses PyTorch; input preparation uses NumPy. No JAX, Haiku or Chex at inference, input preparation or weight conversion.

## Predictors

`foldjet models` prints this list (without loading Torch). Registry: `src/foldjet/models/__init__.py`.

| model id | release |
|---|---|
| `af3` | AlphaFold 3 (DeepMind; weights are access-gated) |
| `protenix1`, `protenix2` | Protenix v1 (`protenix_base_default_v1.0.0`) and v2 (ByteDance) |
| `boltz2` | Boltz-2 |
| `chai1` | Chai-1 (token stream mostly ESM2-3B) |
| `intellifold2` | IntelliFold-v2 (the AF3 graph at wider channels) |
| `openfold3`, `openfold3-preview2` | OpenFold3 v0.5.0 (OpenBind) and the superseded preview-2 |
| `rosettafold3` | RoseTTAFold3 (RosettaCommons foundry) |
| `opendde` | OpenDDE (folds on structural tokens) |
| `esmfold2`, `esmfold2-fast` | ESMFold2 + ESM-C 6B; the Fast variant has 24 trunk blocks and no MSA stack |

Per-family checkpoint evidence and validation scope: [MODEL-INTEGRATION.md](docs/guides/MODEL-INTEGRATION.md).
Against each release's own code, 37 of 38 family x condition rows pass in both modes ([docs/validation/](docs/validation/README.md)).

## Install

One **pixi** environment (torch 2.13.0+cu129, cuequivariance 0.12, FlashAttention-4; `-e cu13` for CUDA 13), Python 3.12.
No GPU is needed to install; on the cluster do it on a CPU allocation, never on a login node.

```bash
git clone git@github.com:CSSB-SNU/foldjet.git
cd foldjet
srun -p h100 -A cssb --qos=normal_h100 --gres=none -c 8 --mem=32G pixi install
source tools/activate_env.sh                   # before running Python directly
```

On A5000 / A6000 / A100 add FlashAttention-2: `FOLDJET_FLASH2=1 sbatch tools/setup_env.sbatch`.

Without pixi (Python 3.12 and a C++20 toolchain; the native AF3 data component is its own wheel):

```bash
python -m pip install ./third_party/af3-data
python -m pip install '.[language]'          # `language`: Transformers for the ESM-C tower
```

**Engine pin.** `pyproject.toml` pins `foldjet-engine==1.0.0` (the renamed miniworld-engine; Python package `foldjet_engine`).
*Stale as of 2026-11-25:* the committed `pixi.lock` still records the pre-rename wheels (`miniworld-engine` 3.0.10 and its
`-kernels-sm90/sm80/sm120-cu129` wheels); refresh the lock before relying on `pixi install --frozen`. The engine changes contracts
between releases: after changing the pin, refresh the lock and re-run the tests.

## Quick start

Weights (download the released blob, then convert it once, offline, to the miniworld checkpoint format):

```bash
foldjet weights download --all-public          # or: foldjet weights download <model>
foldjet weights convert <model>                # writes $FOLDJET_HOME/checkpoints/<model>/<model>.miniworld.pt
```

A fold, on an allocated GPU node (every family takes the same YAML and flags):

```bash
foldjet fold af3 --spec docs/examples/inference/1ubq.yaml --out af3-1ubq
```

From Python:

```python
import foldjet as ff

inputs = ff.load_inputs("input.yaml")
with ff.load_model("boltz2", config="config.yaml", device="cuda:0") as model:
    model(inputs).save("./results/my-run")
    # test-time scaling: keep the trunk, sample diffusion repeatedly
    with model.run_trunk(inputs, recycles=10, seed=0) as trunk:
        for seed in range(10):
            model.sample_diffusion(trunk, samples=5, steps=200, seed=seed).save(f"./results/seed-{seed}")
```

CLI outputs (CIF, confidences and a JSON report of the actual precision, sampling, compile and CUDA-graph settings) always go under
the runs tree `$FOLDJET_RUNS` (default `$FOLDJET_HOME/runs`): `--out NAME` is `<runs>/NAME`, no `--out` gives
`<runs>/<model>/<timestamp-id>/`, and an absolute path outside the runs tree is refused. `result.save(dir)` in Python writes to the
directory you give it. API details: [PYTHON-API.md](docs/guides/PYTHON-API.md).

Other commands: `foldjet ccd prepare|verify` (the CCD LMDB), `foldjet checkpoints` (verify converted blobs against
`checkpoints.lock`), `foldjet validate` (judge a run against the release references). `foldjet` is the same command.

## Backends, precision, modes

- `backend`: `miniworld` (default: per module `cuda` where the engine class has it, else `triton`, else `pytorch`), or one of
  `pytorch | cuda | triton | anthropic | cueq` (`cuequivariance` is an alias). Single ops can be pinned with `engine.ops`;
  `engine.unsupported: fallback | error`. See [BACKENDS.md](docs/guides/BACKENDS.md).
- `precision`: `bf16` (default; fp32 parameters, bf16 autocast = engine `master`) or `fp32`; see [CONFIG.md](docs/guides/CONFIG.md).
- `mode`: `fast` (default; AF3's computation wherever a release's own folds the same) or `exact` (every release convention kept).
- MSA: every predictor keeps up to 16,384 rows and embeds a fresh random 1,024 per trunk pass (the AF3 rule;
  [MINIWORLD-FORMAT.md](docs/guides/MINIWORLD-FORMAT.md)).

## Where things live

```
src/foldjet/        the package: cli.py, api.py, paths.py, models/ (registry, the one AF3 architecture, checkpoints,
                    loading, execution), modules/ (FoldJet-own graph pieces, the engine adapter, ESM-C), diffusion/,
                    data/, eval/, guidance/, train/, utils/
src/foldforge/      compatibility alias of the old name
tests/              pytest suites by responsibility; references/ and release_references/ are frozen ground truth
docs/               user guides, validation verdicts, benchmark records, archive  -> docs/README.md
benchmarks/         paper experiments F1-F11, 5I28 and FoldBench harnesses        -> benchmarks/README.md
tools/              environment, checks, audits, conversion, profiling utilities   -> tools/README.md
scripts/            training drivers (Protenix v1 / AF3, MiniWorld fine-tune)      -> scripts/README.md
third_party/        af3-data: the AF3 feature package (own wheel)
```

Nothing machine-specific lives in the checkout. `$FOLDJET_HOME` (default `~/.cache/foldjet`) holds `checkpoints/`,
`ccd/preprocessed_CCD.lmdb`, `runs/`, `releases/` and `benchmarks/`, each overridable (`FOLDJET_CHECKPOINT_DIR`,
`FOLDJET_CCD_DB`, `FOLDJET_RUNS`, `FOLDJET_RELEASE_ROOT`, `FOLDJET_BENCHMARKS`; `src/foldjet/paths.py`).
Code ownership and where a change goes: [CODE-STRUCTURE.md](docs/guides/CODE-STRUCTURE.md).

## Conventions

- **Never quote a speed number without its conditions**: device, dtype, MSA depth, warm or cold, and which backend *both*
  sides ran. Fold times of the same target and config have differed by 6% across processes from run-to-run variation alone, so
  a before / after taken from two jobs is noise with a sign. A/B in one process, with a same-arm control for the noise floor.
- Tests: `pytest tests` on a compute node (GPU tests skip without a GPU; [tests/README.md](tests/README.md)).
  Lint and types: `bash tools/check_quality.sh` ([CODE-QUALITY.md](docs/guides/CODE-QUALITY.md)).

All documentation: [docs/README.md](docs/README.md).
