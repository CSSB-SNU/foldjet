# Paper experiments (FoldForge side)

Whole-predictor experiments of the {{ENGINE}} / {{FOLDFORGE}} Brief Communication. One folder per experiment with a `README.md` (what is measured, conditions, exact
commands, output columns), its input lists, one script per card and a `collect.py` that turns the raw records into CSVs. This branch (`paper-infer`) holds the
**inference** experiments F3, F4, F5, F9; the others (F1/F2/F8/F11, F6/F7/F10) are on their own branches and add their rows here. The engine's operator experiments
(E1-E6) live in the miniworld-engine repository.

## Common rules

- **One version.** Every experiment of a paper revision runs on one engine release and one FoldForge commit; every result row carries the FoldForge commit, the engine
  version (installed and, when a release-candidate tree shadows it through `ENGINE_SRC` / `PYTHONPATH`, the loaded one), GPU, node and Slurm job id (`common/ffpaper.py`).
- **Cards.** `common/cards/{h100,a100,rtx5090}.env` hold every machine-specific path (checkout, environment, data, caches); each experiment has `h100.sbatch`,
  `a100.sbatch` (cssb3, its own Slurm: partition `A100`) and `rtx5090.sh` (cssb-rtx: no scheduler, the box's GPU lock `~/mwe/gpu.sh`, RAM of everything < 16 GB).
  H100 jobs use `-A cssb --qos=cssb_h100 -p h100`, one GPU, explicit `--mem` (<= 128 GB) and `-c` (<= 32). The environment's `bin/` is put on PATH; never `pixi run` on a
  shared environment; never python on a login node.
- **Outputs** go to `~/foldforge-scratch/paper/<ID>/<card>/` (`PAPER_OUT`), not into the repository; the repository gets the scripts and the final CSVs (`<ID>/data/`).
- **Anthropic kits** are not modified (single exception: the af3_torch featuriser `ref_max_modified_date` line; original `featurise.py.orig_kit`); kit-native modes only
  (`fast`; `big` only for a cell that ran out of memory under `fast`); no other retries. Failures and out-of-memory cells are results: recorded with the reason, never drawn.
- **Timing.** Inference: steady-state median without the model's first call (5I28: forward 1 = cold and forward 2 dropped); 3 repetitions in different jobs where stated.
- **FP8** (H100 only): `MINIWORLD_FP8=safe` + `MINIWORLD_FP8_MODEL=<predictor>`; saturation check (`fp8.check()`) and route counters recorded.
- **Backends.** Engine runs print the per-class backend plan first (`BACKEND_PLAN`) and the served-call table after the first forward (`BACKEND_TABLE`, `all_cuda`); an engine
  class served by Triton or PyTorch in an engine run is a bug, not a result.
- **A walltime run is ONE continuous job**, never chunked into a different job shape; a refused submit stops the script (ask).

## Experiments

| ID | Folder | What | Cards | Paper | State |
|---|---|---|---|---|---|
| F3 | `F3_5i28_times/` | 5I28 1-16 copies: 12 Anthropic kits vs FoldForge bf16 / FP8, seconds per forward | H100, A100, RTX 5090 (FP8: H100; kits: H100, A100) | Fig 1e, Supp 9, Table 6 | scripts smoke-tested; campaign not started |
| F4 | `F4_foldbench27_times/` | 11 predictors x 27 FoldBench targets: engine kernels vs PyTorch operators, time | H100, A100, RTX 5090 | Supp 10 | scripts smoke-tested; campaign not started |
| F5 | `F5_sample_comparison/` | one-to-one samples vs PyTorch (RMSD, pLDDT, PAE) + scores against experimental structures (FoldBench evaluator) | H100, A100, RTX 5090 | Fig 2c-e, Supp 12-15 | scripts smoke-tested; campaign not started |
| F9 | `F9_inference_limit/` | largest 5I28 assembly one forward completes: engine bf16 / FP8, PyTorch operators, Anthropic kit | H100, A100, RTX 5090 | new | scripts smoke-tested; campaign not started |

`common/pairband_check.py` is the check of FoldForge's banded pair conditioning (bit identity banded vs unbanded, and the decision per size) that goes with the pair-band
fix (`DiffusionHead._pair_band_rows`, commit message has the numbers).
