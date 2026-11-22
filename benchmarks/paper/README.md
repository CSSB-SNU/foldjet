# Paper experiments (FoldForge side)

Whole-predictor experiments of the {{ENGINE}} / {{FOLDFORGE}} Brief Communication. One folder per experiment with a `README.md` (what is measured, conditions, exact
commands, output columns), its input lists, one script per card and a `collect.py` that turns the raw records into CSVs. Engine 3.0.6 campaign: inference (F3 F4 F5 F9), FoldBench (F1 F2 F8 F11) and training (F6 F7; F10 removed from the paper) experiments. The engine's operator experiments
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
| F1 | `F1_foldbench_af3/` | AF3 on the whole FoldBench (1,522): GPU-hours of the Anthropic kit vs FoldForge bf16 / FP8 | H100 | Fig 1b, Supp 8 | H100 results of engine 3.0.1 kept (inputs <= 2048 tokens unchanged in 3.0.6) |
| F2 | `F2_foldbench_accuracy/` | per-task accuracy of the F1 runs, top-1 and oracle (upstream rules) | H100 | Fig 2d, Supp 16 | 3.0.1 runs scored, top-1 + oracle |
| F8 | `F8_protenix_ft_foldbench/` | fine-tuned Protenix v1 (10 epochs) vs released, whole FoldBench, top-1 and oracle | H100 | Supp 17 | 2.5.14 runs scored |
| F11 | `F11_fp8_accuracy/` | FP8 vs bf16 accuracy on antibody-antigen / protein-RNA; cause of the antibody-antigen loss | H100 | text | done on 3.0.6: seed-spread design + route ablation (`investigation/`): trunk pair Transition e4m3 rounding; antibody-antigen = trunk bf16, FP8 diffusion only |
| F6 | `F6_training_speed/` | whole-model training speed, AF3 / Protenix v1, 4 backends | H100, A100, RTX 5090 | Fig 1d | H100 3.0.1 kept; A100 / 5090 on 3.0.6 |
| F7 | `F7_finetune_loss/` | Protenix v1 fine-tune loss, engine vs PyTorch (torch.compile) | H100 | Fig 2a,b, Supp 11 | done: same wall-clock (~43 h: engine 1,424 steps, PyTorch 433), user decision |
| F10 | `F10_training_limit/` | training limit, Protenix v1 setup: longest crop; most Pairformer blocks without checkpointing | H100, A100, RTX 5090 | removed | **removed from the paper (user decision, 2026-11-21)**; code and the H100 3.0.6 results kept as a record, no further runs |

`common/pairband_check.py` is the check of FoldForge's banded pair conditioning (bit identity banded vs unbanded, and the decision per size) that goes with the pair-band
fix (`DiffusionHead._pair_band_rows`, commit message has the numbers).

## Engine versions of the measured results (2026-11-22)

FoldForge is pinned to engine **3.0.7**; the campaign rows were measured on **3.0.6** and were not re-measured, because 3.0.7 changes nothing these rows use
(release notes 3.0.7): the H100 Transition change only serves fp32 calls with TF32 off (training, af3_default precision) and the inference runs are bf16; the
bias-logits switch defaults to the 3.0.6 fp32 behaviour (bit-identical); the B200 kernels are not used on H100 / A100 / RTX 5090. Two notes: the
`ParallelResidual` and `bias_only_dit_f32_rows` prebuilt artifacts were rebuilt in 3.0.7 from the same sources (accuracy checked, not compared bit for bit),
so Chai-1 / RoseTTAFold3 times on 3.0.7 may differ within run-to-run noise; the H100 F6 engine row is labelled engine 3.0.7 (see F6 README: measured on
engine main d967378a2 with the fp32 Transition fix). Older rows keep their own `engine_version` column (H100 F1 / F2 / F3 / F4 / F5 / F7 / F8: 3.0.1 - 3.0.2,
unchanged on inputs <= 2,048 tokens).

FoldForge is pinned to engine **3.0.9** (3.0.9 changes only B200 code paths relative to 3.0.8; the A100 / RTX 5090 / H100 campaign rows keep their measured versions 3.0.1 / 3.0.2 / 3.0.6 / 3.0.8 and are not re-measured); the B200 rows were measured on source trees containing the 3.0.9 code (labels in the CSVs). The "pinned to 3.0.7" sentence above is the state of the earlier pin; the 3.0.8 results are kept as they are.

**RTX 5090 coverage.** On the RTX 5090 only ESMFold2 and ESMFold2-Fast are **not feasible** (loading ESM-C 6B, 24 GB fp32 safetensors, exceeds the 5090 host's 16 GB RAM cap; killed by systemd-oomd). The other 9 predictors are measured on the 5090 in F3 / F4 / F5 / F9 (`NOTES_rtx5090_engine306.md`). The Anthropic-kit arms stay "not measured" (user decision), which is a different thing.
