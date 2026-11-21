# F11 investigation: why engine FP8 lowers AlphaFold 3 accuracy on antibody-antigen interfaces

Engine 3.0.6, FoldForge `paper-306`, H100, 2026-11-21. Scripts in `scripts/`, results in `data/`. Raw predictions and evaluator outputs (not in the
repository): `~/foldforge-scratch/paper/F11_investigation/{abl,out,step3}`. 31 GPU jobs, 7.8 GPU-hours.

## Recommendation (used in the paper)

**For antibody-antigen targets, run the trunk in bf16 and keep FP8 in the diffusion module only.** With FP8 restricted to the diffusion module
(token DiT `expand`, atom DiT rows) the antibody-antigen results equal bf16; every trunk FP8 route can stay on for the other tasks, which lose nothing.

```bash
# trunk bf16, diffusion FP8 (engine >= 3.0.2)
export MINIWORLD_FP8=safe                 # the preset: token DiT expand + atom DiT rows stay on
unset  MINIWORLD_FP8_MODEL                # no model -> TriAttn keeps the default fp16 core (no PV / q-k FP8)
export MINIWORLD_TRIMUL_FP8= MINIWORLD_OPM_FP8= MINIWORLD_PWA_FP8= MINIWORLD_TRANSITION_FP8=   # explicitly empty = off (wins over the preset)
```

| config (antibody-antigen, 166 interface groups; seed 42 x 5 samples) | same-sample RMSD to bf16 | mean dDockQ | top-1 % | oracle % | success flips down / up |
|---|---|---|---|---|---|
| bf16 | 0 | 0 | 47.0 | 53.0 | 0 / 0 |
| **FP8 in the diffusion module only (trunk bf16)** | **0.22 A** | **-0.003** | **46.4** | **52.4** | **5 / 3** |
| FP8 `safe` (all routes) | 1.17 A | -0.008 | 43.4 | 47.6 | 63 / 30 |

Protein-protein controls (30 groups): every configuration equals bf16 (top-1 76.7, oracle 80.0; RMSD to bf16 <= 0.19 A).

## Findings

1. **The FP8 perturbation is the same in every task; antibody-antigen is the sensitive one.** Over the whole FoldBench (3.0.1 bf16 / FP8 pair) the
   same-sample deviation follows the model's confidence (ranking score), not the task: antibody-antigen has the lowest confidence (median 0.57 vs 0.77
   protein-protein, 0.89 ligand) and so the largest deviation (median 1.48 A vs 0.74 A); within a confidence bin the two tasks deviate alike.
2. **What moves:** neither the antibody (internal RMSD 0.26 A) nor the antigen (0.3 A) but the antibody's rigid-body pose on the antigen (centroid
   shift median 2.2 A). Correctly docked samples sit in a narrow basin: the same shake moves twice as many out as in (bf16 DockQ > 0.23: FP8 dDockQ
   -0.03 .. -0.08; bf16 DockQ <= 0.1: +0.02), so the net is a loss. Fab and single-chain antibodies lose alike; no relation to size, templates or chain count.
3. **Route:** the trunk pair Transition (D 128, n 4; 48 blocks x 12 passes). Transition alone in FP8 reproduces the whole `safe` loss; each other route
   alone (TriMul, OPM, PWA, TriAttn PV + q/k, the diffusion routes) stays near bf16; together they add a small part (`safe` without Transition still
   loses: net flips -23 vs -33). The diffusion conditioning transitions in FP8 are harmless (RMSD 0.12 A, 0 flips down).
4. **Cause inside the Transition: e4m3 rounding (3 mantissa bits), not the scales.** h headroom 1 or 256, per-call (dynamic) rescaling and static
   calibration all keep the loss; FP8 on only the expand GEMM or only the squeeze GEMM keeps it too.
5. **Mechanism:** the inter-chain pair entries are small (rms ~80-125, last block 7-28, against a pair rms of 1,200-3,300); the same absolute FP8 error
   is a large relative error there and grows over the blocks (inter-chain relative error 3-9 % at block 0, 13-51 % at block 47; the whole pair 0.5 ->
   1.6 %). The pattern is the same for antibody-antigen and protein-protein; the low-confidence antibody-antigen pose turns it into a docking change.

Confidence: route identification high (single-route reproduction, the others harmless, consistent Transition variants); mechanism medium (one seed x 5
samples per configuration, wide DockQ intervals; the net flip count is negative in every lossy configuration).

## Files

| file | content |
|---|---|
| `data/abl_table.txt`, `data/abl_summary.csv`, `data/abl_ci_abag.csv` | route ablation: every configuration, antibody-antigen and protein-protein, with bootstrap intervals |
| `data/abag_report.txt`, `data/abag_per_*.csv`, `data/abag_target_attrs.csv` | where the deviation sits (antigen-superposed per-region deviations, pose shift), dDockQ by bf16 DockQ bin, correlations |
| `data/wholefb_dprec_dseed.csv`, `data/tasks_*.csv`, `data/conf_vs_dprec.csv` | whole-FoldBench deviation per task and per confidence bin |
| `data/step3/<target>/` | per-block relative error of the pair representation (Transition-only FP8 vs bf16) and the FP8 Transition calls |
| `data/targets*.txt`, `abag.txt`, `pp25.txt` | the ablation subset (113 antibody-antigen + 25 protein-protein targets) |
| `scripts/` | runners (`abl.sbatch`, `abl_eval.sh`), analyses (`step1.py`, `abag_report.py`, `abl_report.py`, `conf_vs_dprec.py`), `overlay/sitecustomize.py` (TriAttn kind / Transition variant switches used only by this investigation) |
