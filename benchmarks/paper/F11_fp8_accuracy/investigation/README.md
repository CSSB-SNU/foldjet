# F11 investigation: why engine FP8 lowers AlphaFold 3 accuracy on antibody-antigen interfaces

Engine 3.0.6, FoldJet `paper-306`, H100, 2026-11-21. Scripts in `scripts/`, results in `data/`. Raw predictions and evaluator outputs (not in the
repository): `~/foldforge-scratch/paper/F11_investigation/{abl,out,step3}`. 31 GPU jobs, 7.8 GPU-hours.

## Recommendation (used in the paper)

**For antibody-antigen targets, run the trunk in bf16 and keep FP8 in the diffusion module only.** With FP8 restricted to the diffusion module
(token DiT `expand`, atom DiT rows) the antibody-antigen results equal bf16; every trunk FP8 route can stay on for the other tasks, which lose nothing.

```bash
# trunk bf16, diffusion FP8 (engine >= 3.0.2)
export FOLDJET_ENGINE_FP8=safe                 # the preset: token DiT expand + atom DiT rows stay on
unset  FOLDJET_ENGINE_FP8_MODEL                # no model -> TriAttn keeps the default fp16 core (no PV / q-k FP8)
export FOLDJET_ENGINE_TRIMUL_FP8= FOLDJET_ENGINE_OPM_FP8= FOLDJET_ENGINE_PWA_FP8= FOLDJET_ENGINE_TRANSITION_FP8=   # explicitly empty = off (wins over the preset)
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

## Where the error grows: per-pass, per-block trace (2026-11-22)

Transition-only FP8 against bf16 in one process per target (engine 3.0.6, H100, eager, seed 42, 11 passes x 48 blocks; the 4 MSA-module pair
Transitions are FP8 too). Relative error of the pair after every pair module of every block and pass, per region (inter = antibody x antigen, ab, ag,
all; ~64 evenly spaced tokens per region), and of the single. Checks: the FP8 kernel on an extracted input slice equals the in-model output bit for bit;
a repeat run is bit-identical. Targets: antibody-antigen with a loss under Transition-only FP8 (8f5n, 8a44, 8be2, 8w84; dDockQ -0.21 .. -0.75),
antibody-antigen without a loss (7xld, 8aom, 8c3l, 8e0e), protein-protein (7spq, 8a51, 8upb). H100 jobs 65140-65143 (0.53 GPU-h).

1. **The split is per recycle pass, not per block.** In pass 0 the three groups are the same (inter error 0.7 % at block 0, 1.5-2.3 % by block 40; the
   jump at block 47 to 10-40 % is a small-denominator effect in every group). After passes 1-2 every target is at 2-4 %; the no-loss antibody-antigen
   and the protein-protein targets stay there up to pass 10 (max 4.5 %), the loss targets keep growing: > 5 % from pass 5 (8a44, 8be2), pass 6 (8f5n),
   pass 9 (8w84). The single follows (no loss / pp flat at 0.5 %; loss targets > 1 % from pass 5, 1.8-2.0 % at pass 10).
2. **Injection is the same, the retention differs.** One FP8 Transition call adds the same error in every group (0.29-0.32 % of x, 6.7-6.9 % of the
   update, inter), and the summed injected energy over 11 passes is alike (125-191). The final error energy over the injected energy is 0.03-0.09 for
   no-loss and protein-protein, 0.11 / 0.21 / 0.87 / 1.90 for the loss targets (8f5n / 8w84 / 8be2 / 8a44). Within a pass the triangle
   multiplications grow the error slightly (incoming x1.010-1.016 per module, outgoing x1.003-1.009), the Transition shrinks it (x0.97-0.98); the pass
   boundary (recycle + embedder + MSA module) multiplies it by ~1.15 in every group. No-loss targets bring it back down over the next pass; loss targets
   carry it on.
3. **No difference in magnitudes or e4m3 use.** Pair rms (inter 118 / 121 / 125), the subnormal share of the scaled h (28-33 %) and the h quantisation
   error (2.68-2.70 %) are the same in the three groups; a per-row scale lowers the h error by 0.04 points only.
4. **Schedules** (final-pass trunk inter error, threshold: <= 15 % no pose change, >= 24 % pose change in the downstream predictions): FP8 only in
   blocks 0-23, only in pass 0, only TriMul FP8, per-row scale emulation or protecting the inter-chain region in bf16 all leave at least one loss target
   collapsed (8a44 at 68-72 % in each). Only "bf16 in passes 0-9, FP8 in the last pass" brings all four loss targets to 9-12 %, which keeps 1/11 of
   the FP8 Transition calls (no speed gain left). The practical rule stays the recommendation above: trunk bf16 for antibody-antigen.
5. **Recycle adds in fp32 do not help.** Rerun with the pair and single recycle adds (LayerNorm, projection, add) in fp32, one rounding to bf16,
   reference included (`trace.py --rec32`, jobs 65177-65179): loss targets 76.3 / 65.2 / 52.9 / 21.1 % (8f5n / 8a44 / 8be2 / 8w84; bf16 recycle 12.5 /
   60.6 / 50.6 / 24.3), no-loss and protein-protein unchanged (9.5-14.9 %). The bf16 rounding of the recycle is not what carries the error over;
   the trunk's response on these targets is.

Reading: the FP8 error is the same everywhere; whether the recycle loop damps it depends on the target. Targets with a narrow docking basin do not damp
it (8f5n is bistable: the same configuration ends at ~12 % or ~77 % in near-identical variants), and the same targets collapse when the error comes
from another route (8a44 under TriMul-only FP8: dDockQ -0.62). The four no-loss antibody-antigen targets behave like protein-protein. Limits: one seed,
11 targets, subsampled tokens, decisions on trunk error only (the schedules and the fp32 recycle were not run through diffusion / DockQ).
Files: `scripts/trace/` (trace.py, analyze.py, down.py and their sbatch), `data/trace/report_raw.txt` (all tables), `data/trace/sched_last.csv`,
`data/trace/downstream_join.csv`, `data/trace_rec32/` (the fp32-recycle run). Raw traces: `~/foldforge-scratch/paper/F11_investigation/trace/out*/`.

## Files

| file | content |
|---|---|
| `data/abl_table.txt`, `data/abl_summary.csv`, `data/abl_ci_abag.csv` | route ablation: every configuration, antibody-antigen and protein-protein, with bootstrap intervals |
| `data/abag_report.txt`, `data/abag_per_*.csv`, `data/abag_target_attrs.csv` | where the deviation sits (antigen-superposed per-region deviations, pose shift), dDockQ by bf16 DockQ bin, correlations |
| `data/wholefb_dprec_dseed.csv`, `data/tasks_*.csv`, `data/conf_vs_dprec.csv` | whole-FoldBench deviation per task and per confidence bin |
| `data/step3/<target>/` | per-block relative error of the pair representation (Transition-only FP8 vs bf16) and the FP8 Transition calls |
| `data/targets*.txt`, `abag.txt`, `pp25.txt` | the ablation subset (113 antibody-antigen + 25 protein-protein targets) |
| `scripts/` | runners (`abl.sbatch`, `abl_eval.sh`), analyses (`step1.py`, `abag_report.py`, `abl_report.py`, `conf_vs_dprec.py`), `overlay/sitecustomize.py` (TriAttn kind / Transition variant switches used only by this investigation) |
