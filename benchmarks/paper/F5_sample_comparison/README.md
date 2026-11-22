# F5: one-to-one comparison with PyTorch, and scores against the experimental structures

Paper: Fig 2c-e, Supp 12-15. Uses the F4 runs (same inputs, same seeds, 5 samples per target): no extra GPU work. Cards: H100, A100, RTX 5090 (the CPU steps run on
the H100 cluster; predictions of the other cards are copied there first).

## What is measured

**(a) Sample against sample** (`pair.py`): for every predictor, target and kernel arm (`engine`, `engine_fp8`), sample k against sample k of the PyTorch arm (same input,
same seed): superposed Ca / C1' RMSD (A), mean |dpLDDT| per atom, mean |dPAE| per token pair (+ the signed means). **Baseline** (arm `baseline`): the same measures between
different samples k < l of the PyTorch arm = how much two draws of the same model differ anyway.

**(b) Against the experimental structure** (`eval.sbatch`, CPU): the upstream FoldBench evaluator (OpenStructure through its singularity image, DockQ v2) on the 5 samples;
per (predictor, target / interface) the mean over the 5 samples of lDDT (monomers), DockQ (protein interfaces, DockQ v2 for protein-DNA / RNA) and lDDT-PLI (protein-ligand),
for `torch`, `engine`, `engine_fp8`; the kernel arm minus PyTorch and whether the difference is inside PyTorch's own s.d. over its 5 samples. `evaluate_ranked.py` is the
paper's wrapper (prediction reference table with the mean-pLDDT ranking score, the `ost` shim, task CSVs restricted to the predicted targets). Upstream's `best`
(oracle) summary is not needed here and may fail on monomer GDT-TS (upstream bug, noted in the wrapper).

ESMFold2 on RNA / DNA-only targets: see F4 (`allow_no_protein` is not in `fdad9e2`); those targets have no prediction and drop out of the comparison.

## Commands

```bash
# A100 / 5090 only: bring the F4 outputs to the H100 cluster
rsync -a cssb3:foldforge-scratch/paper/F4/a100/ ~/foldforge-scratch/paper/F4/a100/
rsync -a cssb-rtx:foldforge-scratch/paper/F4/rtx5090/ ~/foldforge-scratch/paper/F4/rtx5090/
# (b) scores: one (predictor, arm) per array task of eval_items.txt (33 tasks), CPU
RUNS=~/foldforge-scratch/paper/F4/h100 CARD=h100 sbatch --array=0-32 benchmarks/paper/F5_sample_comparison/eval.sbatch
# (a) + CSVs, after (b)
RUNS=~/foldforge-scratch/paper/F4/h100 CARD=h100 sbatch benchmarks/paper/F5_sample_comparison/pair.sbatch
```

Smoke: after the F4 smoke of two targets, `ITEMS=<file with "af3 torch" and "af3 engine"> RUNS=... CARD=h100 sbatch --array=0-1 .../eval.sbatch`, then `pair.sbatch` with `MODELS=af3`.
Expected wall time (H100 cluster CPU): eval ~10-40 min per task (OpenStructure on 27 targets x 5 samples), `pair.py` ~10-20 min.

## Output (`data/`)

| file | content |
|---|---|
| `F5_<card>_pairs_summary.csv` | per (arm, predictor) and `ALL`: n sample pairs, median / p90 RMSD, median \|dpLDDT\|, median \|dPAE\| (arm `baseline` = PyTorch vs PyTorch); each row: card, engine version, FoldForge commit |
| `F5_<card>_gt.csv` | per (predictor, task, key): mean of 5 samples per arm, PyTorch's s.d., arm - torch |
| `F5_<card>_gt_summary.csv` | per arm and kind (monomer / interface / ligand / all): n, median \|diff\|, mean signed diff, fraction within PyTorch's 1 s.d., r |

The per-sample-pair table (`pairs.csv`, the shape of the paper's `pair_compare.csv`) stays in `~/foldforge-scratch/paper/F5/<card>/`.

## A100 / RTX 5090: OpenDDE engine arm on engine 3.0.8 (2026-11-22)

The OpenDDE engine predictions of these two cards are the engine-3.0.8 F4 re-run (TokenDiTProtenix `extra_pair_bias` on cuda; see the F4 README); the
pairs were recomputed (jobs 65196 A100, 65197 RTX 5090; every other predictor unchanged, `engine_version` column `3.0.6;3.0.8`). OpenDDE, engine vs PyTorch
sample k vs sample k, median Cα/C1' RMSD: A100 0.135 -> 0.271 A (80 pairs), RTX 5090 0.152 -> 0.265 A (65 pairs; 7wvs has no pair on either version, the
PyTorch arm is out of memory there); the PyTorch sample-to-sample baseline is 1.25 / 1.20 A. All predictors: A100 engine median 0.303 -> 0.316 A, RTX 5090
0.287 -> 0.303 A. The engine-3.0.6 OpenDDE predictions and the previous pairs are kept next to the runs (`archive_306/`).
