# F11: FP8 accuracy on antibody-antigen and protein-RNA (AF3, FoldForge)

**Question (user, 2026-11-21).** Not whether a bf16 re-run differs less than FP8 from bf16 (it trivially does), but whether the FP8 (low-precision)
perturbation of the **same seed** is small compared with the natural variation **between seeds**: is running FP8 like choosing another seed, or a real
precision loss?

## Data

The 162 targets of the two tasks (`targets_abag_rna.txt`: 113 antibody-antigen, 49 protein-RNA), AF3 (DeepMind weights), 5 trunk seeds (42, 66, 101,
2024, 8888) x 5 samples, 200 steps, 11 trunk passes, compile + CUDA graphs, the same seeds and sample indices in both arms.

| pair | engine | bf16 run | FP8 run (`MINIWORLD_FP8=safe`, `MINIWORLD_FP8_MODEL=af3`) |
|---|---|---|---|
| primary `v306` | 3.0.6 (FoldForge `paper-306`) | `~/foldforge-scratch/paper/F11/h100/af3_306_bf16_abagrna` (Slurm 62832) | `af3_306_fp8_abagrna` (62834; `fp8.check()` after every group, 0 saturations) |
| replicate `v301` | 3.0.1 (FoldForge a9e1549) | `~/foldforge-scratch/foldbench/ff_full/af3_301` (58554, whole FoldBench) | `af3_301_fp8` (58867, no `check()`) |

Per-sample scores: the FoldBench evaluator (F2 `evaluate.py`, 25 samples; Slurm 62944 / 62945 for 3.0.6, the F11 evaluations of 3.0.1). Upstream rules:
interface groups (pdb_id, chain 1, chain 2), rows with a NaN DockQ dropped (166 of the 172 antibody-antigen groups remain), success = DockQ >= 0.23.

## Design (`analyze_seed.py`, CPU only)

**A. Structure** (C-alpha / C1' RMSD after superposition, per target):
`d_prec(s,k)` = FP8 seed s sample k vs bf16 seed s sample k; `d_seed` = bf16 seed s sample k vs bf16 seed s' sample k (s != s'); `d_sample` = bf16 seed s
sample k vs sample k' of the same seed (context). Per task: medians, the per-target ratio median d_prec / median d_seed (median, IQR), the fraction of
targets with median d_prec < median d_seed, and the fraction of (target, s, k) whose FP8 twin is closer to bf16 (s, k) than the nearest bf16 sample of any
other seed.

**B. Accuracy, each seed = a replicate of 5 samples.** Per seed and arm: success of the seed's own top-1 (largest ranking score among its 5 samples), the seed's
oracle (best of its 5), the mean DockQ of its 5. Between-seed spread (bf16): SD and range over the 5 seeds, and |m(bf16 s) - m(bf16 s')| over the 10 seed
pairs. Precision effect: m(FP8 s) - m(bf16 s), paired per seed, averaged over seeds. 95% intervals by bootstrap over interface groups (5,000 resamples).
Verdict per task and metric: **inside** the between-seed spread when the precision effect's interval includes 0 and |effect| <= the mean between-seed
|difference|; **outside** (a precision loss) when the interval excludes 0 and |effect| > the mean between-seed difference; otherwise **borderline**.

## Commands

```bash
E=~/foldforge-scratch/ff_2512/.pixi/envs/default; O=~/foldforge-scratch/paper/F11/h100/analysis_seed
sbatch -p h100 -A cssb --qos=cssb_h100 --gres=none -c 16 --mem=48G --time=02:00:00 -J F11seed --wrap \
  "cd benchmarks/paper/F11_fp8_accuracy; OMP_NUM_THREADS=1 $E/bin/python analyze_seed.py structure --out $O --workers 16 && $E/bin/python analyze_seed.py accuracy --out $O"
```
(Slurm 63099, 2 min 17 s.) The earlier run-to-run analysis (`analyze.py`, bf16 3.0.6 vs bf16 3.0.1 etc.) is kept as a sanity check.

## Outputs (`data/engine306/`, every row has `engine_version`)

| file | content |
|---|---|
| `f11seed_structure_summary.csv` | per pair and task: median d_prec / d_seed / d_sample, ratio median + IQR, fraction of targets d_prec < d_seed, fraction of samples whose FP8 twin is closer than any other-seed sample |
| `f11seed_structure_per_target.csv`, `f11seed_structure_samples.csv` | the per-target and per-(target, seed, sample) values |
| `f11seed_accuracy_summary.csv` | per pair, task, metric: the 5 per-seed values of each arm, bf16 SD / range / seed-pair differences, precision effect with interval, between-seed |difference| with interval, verdict |
| `f11seed_accuracy_per_group_seed.csv` | per interface group and seed: top-1 / oracle success and mean DockQ, both arms |
| `f11seed_verdict.txt` | the verdict text |
| `f11_verdict_runtorun.txt`, `f11_*.csv` | the earlier run-to-run analysis (sanity: bf16 3.0.6 vs bf16 3.0.1 same-sample RMSD median 0.03 / 0.08 A, top-1 success changes 1.7% of groups) |

## Result (engine 3.0.6 primary; 3.0.1 replicate in the CSV)

- **Structure:** the FP8 perturbation of a sample is 4-7x smaller than the change from another seed (median d_prec / d_seed per target 0.23 for
  antibody-antigen, 0.33 for protein-RNA); for 96% / 90% of the targets median d_prec < median d_seed; but only 65% / 68% of the FP8 samples are closer to
  their bf16 twin than to the nearest sample of any other seed.
- **Accuracy:** protein-RNA: every metric inside the between-seed spread (FP8 ~ another seed). Antibody-antigen: oracle (-4.0 pt [-6.3, -1.7] vs between-seed
  2.8 pt) and mean DockQ of the samples (-0.017 [-0.025, -0.008] vs 0.014) **outside** (a precision loss); top-1 (-2.4 pt [-4.8, -0.1] vs 4.1 pt) borderline.
  The 3.0.1 replicate: antibody-antigen top-1 / oracle inside, mean DockQ borderline (-0.011 [-0.022, -0.000]); protein-RNA inside.

## Investigation: the antibody-antigen FP8 loss (2026-11-21)

Route ablation and mechanism in [`investigation/README.md`](investigation/README.md). Cause: the trunk pair Transition in FP8 (e4m3 rounding, not the
scales). **Recommendation used in the paper: for antibody-antigen targets run the trunk in bf16 and FP8 only in the diffusion module** (the antibody-antigen
results then equal bf16; the other tasks lose nothing with every route on).
