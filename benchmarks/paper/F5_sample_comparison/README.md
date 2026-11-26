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
| `F5_<card>_pairs_summary.csv` | per (arm, predictor) and `ALL`: n sample pairs, median / p90 RMSD, median \|dpLDDT\|, median \|dPAE\| (arm `baseline` = PyTorch vs PyTorch); each row: card, engine version, FoldJet commit |
| `F5_<card>_gt.csv` | per (predictor, task, key): mean of 5 samples per arm, PyTorch's s.d., arm - torch |
| `F5_<card>_gt_summary.csv` | per arm and kind (monomer / interface / ligand / all): n, median \|diff\|, mean signed diff, fraction within PyTorch's 1 s.d., r |

The per-sample-pair table (`pairs.csv`, the shape of the paper's `pair_compare.csv`) stays in `~/foldforge-scratch/paper/F5/<card>/`.

## A100 / RTX 5090: OpenDDE engine arm on engine 3.0.8 (2026-11-22)

The OpenDDE engine predictions of these two cards are the engine-3.0.8 F4 re-run (TokenDiTProtenix `extra_pair_bias` on cuda; see the F4 README); the
pairs were recomputed (jobs 65196 A100, 65197 RTX 5090; every other predictor unchanged, `engine_version` column `3.0.6;3.0.8`). OpenDDE, engine vs PyTorch
sample k vs sample k, median Cα/C1' RMSD: A100 0.135 -> 0.271 A (80 pairs), RTX 5090 0.152 -> 0.265 A (65 pairs; 7wvs has no pair on either version, the
PyTorch arm is out of memory there); the PyTorch sample-to-sample baseline is 1.25 / 1.20 A. All predictors: A100 engine median 0.303 -> 0.316 A, RTX 5090
0.287 -> 0.303 A. The engine-3.0.6 OpenDDE predictions and the previous pairs are kept next to the runs (`archive_306/`).

## B200 data (DATA_STATUS, 2026-11-22)

Host `b200` (8 x B200 sm_100a, 178.35 GiB each, shared account, no Slurm), campaign 2026-11-22. FoldJet `paper-b200` (base origin/main 1688f89), engine run from SOURCE (the PyPI wheels carry no sm100 kernels) with the sm100a prebuilt JIT cache. Environment: venv `ffenv` over the engine team's pixi env eng-2511 (torch 2.13.0+cu129, triton 3.7.1, cuequivariance 0.12); `transformer_engine` blocked in `ffenv` (sitecustomize) so ESMC takes the same PyTorch fallback as on the other cards. Weights, FoldBench inputs (27-target subset of the 1,532 copied; `data.yaml` paths rewritten to the B200 copy) and 5I28 LMDBs copied from the H100 cluster.
Execution: 4 always-ours lanes (GPUs 0-3) + GPUs 4-7 only while no foreign process was on them (the lane watches `nvidia-smi` compute apps; a task that saw a foreign process is `external-contention`: output purged and re-queued, never a result; a lane on 4-7 stops at the first foreign process and is not restarted). One task per GPU at a time (no overlapping on one GPU); concurrent lanes share the node (host memory bandwidth / power), no correction applied. Driver: `common/b200_lane4.sh`, tasks of `common/queue_b200_v2.txt` through `common/b200_task.sh`, `.done` marker only when the task's own validity check passed.
Engine labels: **A** = `engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)`: F6 engine arm and F4 OpenDDE engine arm. **B** = `engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)`: every other engine arm. Both are in the CSV column `engine_label` (and `notes`). The CSV column `engine*_version_installed` says 2.5.11 because it is the package metadata of the shared env; the loaded source is in `*_engine_version_loaded` / `engine_loaded_from`.
B200 TriAttn: the released 3.0.7 B200 TriangleAttention kernels (`triattn_sm100.cu`, `triattn_dh64_sm100.cu`, `tri_l128_sm100.cu`, `ex2_poly2` without an upper clamp) produced non-finite output on real AF3 / Chai-1 inputs; those first-pass engine records (F4: AF3 21 / 27, Chai-1 25 / 27 non-finite; F3 AF3 n >= 2, Boltz2 n >= 12 ...) are NOT used (`paper/invalid_archive/` on the B200, not in the CSVs).
Backend report (engine arms): LayerNorm of 65,536 (F4 / F3 / F9) or 589,824 (F6) rows of width 64 in bf16 is refused by the engine's B200 CUDA route by design (Triton is faster there, measured; user-declared LayerNorm exception) and served by Triton: 8 calls per forward (F4 / F3), 9 per micro step (F6). Everything else is served by cuda; ESMFold2, IntelliFold2 and RoseTTAFold3 have no such call (`engine_cuda_only = yes`). So the B200 engine arm is not 'Triton 0': the only Triton calls are the LayerNorm w64 exception (`F4_backend_engine_b200.csv`).

**F5 (this folder), part (a) only:** `data/F5_b200_pairs_summary.csv` = sample k against sample k of the PyTorch arm (same input, same seed) on the B200 F4 predictions (arm `engine`; no `engine_fp8`), 1,340 engine pairs (+ 2,680 PyTorch-vs-PyTorch baseline pairs): engine median C-alpha / C1' RMSD 0.351 A (p90 3.28), median |dpLDDT| 0.213, median |dPAE| 0.117; PyTorch-vs-PyTorch baseline 1.468 A (p90 17.5), 0.348, 0.289. Per-predictor rows in the CSV (engine median RMSD 0.16 A RoseTTAFold3 .. 0.60 A ESMFold2 / Protenix v1). The 4,020-row per-pair table stays in `~/foldforge-scratch/paper/F5/b200/pairs.csv` (H100 cluster); the B200 F4 predictions were copied to `~/foldforge-scratch/paper/F4/b200/` (26 GB) and the pairs were computed by `pair.py` + `collect.py` on the cluster CPU (`srun --gres=none`). OpenDDE pairs are the engine-label-A arm (holes fix); all others label B. **Not run:** (b) the FoldBench-evaluator scoring against the experimental structures (as for the RTX 5090; the predictions are in `F4/b200/pred/`). The CSV `engine_version` column is a free-text label, the engine labels are in the F4 CSV.
Engine 3.0.9 (PyPI, tag v3.0.9, 2026-11-22) contains the code of both labels (B200 TriAttn exp-clamp, fp32 Transition `ieee_f32_sm100`, `extra_pair_bias` on the B200 composed runner); the B200 rows were measured on the source trees named above, not on a 3.0.9 install (there is no B200 kernels wheel).

## B200: ESMFold2 at the release sampler (134 steps) (2026-11-23)

`data/F5_b200_esmfold2_134_pairs_summary.csv` (part (a)) on the predictions of the F4 B200 ESMFold2 134-step runs: engine sample k vs PyTorch
sample k median Ca/C1' RMSD 0.658 A (p90 2.66, 135 pairs, now including the nucleic-acid-only targets); PyTorch sample-to-sample baseline
2.663 A (270 pairs); median |dpLDDT| 0.51 vs 1.09. Replaces the ESMFold2 rows of `data/F5_b200_pairs_summary.csv` (200 steps: 0.604 A, 105 pairs).

## RTX 5090: engine 3.0.10 bf16 (measured on branch 5090-int) + engine FP8, sample pairs and FoldBench scores (DATA_STATUS, 2026-11-24)

**Released as 3.0.10 (2026-11-24):** engine `5090-int` 07bf36ae4 -> v3.0.10 changes no code the RTX 5090 runs (TokenDiTMiniWorldBiasOnly `backend_versions` number, the GPU-less prebuild plan, an A100-only `MWE_SM120` guard in `wide_ops.cu`), so these rows are the 3.0.10 rows; the CSV `engine` columns say so and keep the measured commit.

Predictions of the F4 runs of the same date (see the F4 README): engine bf16 on cssb-rtx (real weights, the old rows' box), engine FP8 on the rented
4x RTX 5090 (`vast5090`, real weights; `FOLDJET_ENGINE_FP8=safe` + `FOLDJET_ENGINE_FP8_MODEL`), engine branch `5090-int` 07bf36ae4 (= 3.0.10 on sm_120), FoldJet
49d41bb. The PyTorch reference = the existing 5090 PyTorch-arm predictions of the 3.0.6 campaign (not re-run: same inputs and seeds, and FoldJet
6a0593e -> 49d41bb changed only ESMFold2 code). `data/F5_rtx5090_pairs_summary.csv` (released engine) is unchanged.

- `data/F5_rtx5090_3010_pairs_summary.csv`: part (a) for arms `engine` and `engine_fp8` + the PyTorch baseline, with the released rows
  side by side (`*_prev`, `rmsd_median_new_over_prev`; there is no previous 5090 FP8 arm). Pairs: `pair.py` per predictor on the cluster CPU
  (10-minute jobs) for bf16; for FP8 on the rented node's CPU (the 7 GB of PAE arrays were not copied back; the PyTorch predictions were copied there).
- `data/F5_rtx5090_3010_gt.csv`, `..._gt_summary.csv`: part (b), **the first FoldBench-evaluator scoring of 5090 predictions** (PyTorch,
  engine, FP8 arms), cluster CPU in 10-minute pieces (one (predictor, arm, FoldBench task) per job: `evaluate_ranked.py --targets <task>`, the
  per-task raw tables merged afterwards; FP8 from a copy holding the CIFs and pLDDT only).

| arm | pairs | median RMSD to PyTorch sample k (A) | p90 | median abs dpLDDT | previous (released) median RMSD |
|---|---|---|---|---|---|
| engine bf16 | 890 | 0.282 | 2.27 | 0.178 | 0.303 |
| engine FP8 | 890 | 0.307 | 2.71 | 0.191 | (none) |
| PyTorch vs PyTorch (baseline) | 1,780 | 1.313 | 15.10 | 0.260 | 1.313 |

Same-sample agreement with PyTorch stays at the old level (bf16 median 0.282 vs 0.303 A on the released engine; per predictor 0.10-0.47 A, ratios
new / old 0.84-1.09); FP8 0.31 A (per predictor 0.11-0.48 A), both far inside the PyTorch sample-to-sample spread (1.31 A).

Scores against the experimental structures (mean of the 5 samples per target / interface; arm minus PyTorch):

| arm | kind | n | median abs diff | mean signed diff | within PyTorch's 1 s.d. | r |
|---|---|---|---|---|---|---|
| engine bf16 | all | 197 | 0.0037 | +0.0027 | 90.4 % | 0.9971 |
| engine FP8 | all | 197 | 0.0042 | +0.0018 | 86.3 % | 0.9885 |
| engine FP8 | monomer | 80 | 0.0052 | -0.0057 | 81.3 % | 0.9932 |

(H100 for reference, `data/existing/F5_h100_engine3.0.1_gt_summary.csv`: engine 0.0034 / 91.8 %, FP8 v2 0.0047 / 83.9 %.)
