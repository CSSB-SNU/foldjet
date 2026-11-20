# Paper experiments (FoldForge side): benchmarks/paper

Whole-predictor experiments of the paper, one folder per experiment (README with what is measured, conditions and commands; input lists; per-card run scripts;
`collect.py` -> CSV). Run outputs go to `$PAPER_OUT/<ID>/<card>/` (default `~/foldforge-scratch/paper/`), never into the repository; the repository holds the scripts and the final CSVs.
All experiments of a paper version are measured on ONE engine release and ONE FoldForge commit; every result carries `stamp.json` (commit, engine, GPU, node, Slurm job id).

| ID | folder | paper | content | card | state |
|---|---|---|---|---|---|
| F1 | `F1_foldbench_af3/` | Fig 1b, Supp 8 | AF3 on the whole FoldBench (1,522): GPU-hours of the Anthropic kit vs FoldForge bf16 / FP8 | H100 | scripts smoke-tested on the engine 3.0.6 candidate; campaign not run (after the release) |
| F2 | `F2_foldbench_accuracy/` | Supp 16 | per-task accuracy of the F1 runs, top-1 and oracle (FoldBench evaluator, upstream rules) | H100 | evaluator + scoring used for F11 on real data; campaign after F1 |
| F8 | `F8_protenix_ft_foldbench/` | Supp 17 | fine-tuned Protenix v1 (10 epochs) vs released, whole FoldBench, top-1 and oracle | H100 | scripts written, smoke-tested through F1's runner; campaign not run |
| F11 | `F11_fp8_accuracy/` | text | FP8 vs bf16 accuracy on antibody-antigen / protein-RNA: rounding or reordering | H100 | analysis of runs 58554 / 58867 / 58996 done (CPU); re-measure with the released engine after the campaign |

F3-F7, F9, F10 are on other branches. Conventions: H100 jobs `--qos=cssb_h100`; CPU jobs `--gres=none`; no python on a login node; never `pixi run` on a shared env;
failures / OOM are results (recorded, not drawn); FP8 = `MINIWORLD_FP8=safe` + `MINIWORLD_FP8_MODEL=<model>`, saturation checks recorded (`run.json` `fp8`).
`common/` holds the shared defaults (`env.sh`) and the run stamp (`stamp.py`).
