# F2: accuracy per FoldBench task of the F1 runs (top-1 and oracle)

**What is measured.** The structures of the F1 runs (kit, FoldForge bf16, FoldForge FP8; 25 samples per target = 5 trunk seeds x 5 samples) scored against the experimental
structures with the **upstream FoldBench evaluator** (OpenStructure lDDT / DockQ v2, run unchanged), and aggregated per task with the **rules of upstream `task_score_summary.py`**:

* **top-1** = per target (per interface for the interface tasks) the sample with the largest ranking score (AF3: 0.8 ipTM + 0.2 pTM, pTM alone for one chain; the kit's own `ranking_score`);
  the first sample in file order on a tie (upstream `idxmax`);
* **oracle** = per target (interface) the best of the 25 samples by the task metric (largest DockQ; smallest rmsd for protein-ligand; largest lDDT for monomers);
* interfaces: success = DockQ >= 0.23 (protein-DNA / protein-RNA from the `*_dockqv2.csv` table); protein-ligand: rmsd < 2 A and lDDT-PLI > 0.8; monomers: mean lDDT (x 100);
* rows whose metric is NaN are dropped before the selection, the rate is over the selected rows (upstream's `df[df[metric].notna()]`).

`score_tasks.py` implements these rules (upstream's own `--metric_type best` aborts on the monomer GDT-TS column, so the oracle cannot be taken from upstream's script);
`score_tasks.py --check-upstream` compares its top-1 column with upstream's own `summary_*_rank.csv` (the evaluator run also writes that summary), which is the test of this file.

## Files

| file | what |
|---|---|
| `evaluate.py` | FoldBench evaluation of one run (several `--pred-dir` for a shard-mode run): `prediction_reference.csv` from the layout + `ranking.csv`, task CSVs restricted to the predicted targets, upstream `evaluate.py` + `task_score_summary.py` unchanged, then `scores_<name>.csv` (top-1, oracle) |
| `score_tasks.py` | the selection rules above, from the evaluator's raw per-sample tables; also `per_target()` (selected sample per target) used by F11 |
| `eval.sbatch` | CPU Slurm job (`--gres=none`, 32 CPUs, 120 GB) around `evaluate.py` |
| `collect.py` | `scores_*.csv` of several arms -> `f2_task_scores.csv` (long, with the run stamps), `f2_top1_wide.csv`, `f2_oracle_wide.csv` (the shape of the paper's `tables/foldbench_full_af3_accuracy.csv`) |
| `results/` | the CSVs of the campaign |

## Environment

* FoldBench data: `FB_ROOT` (default `~/data/foldbench`: `targets/`, `ground_truth/`, `upstream/` and `upstream_cy/`; `upstream_cy` is the upstream copy with the DockQ v2 Cython
  operations built for Linux, ~10x faster than the pure-Python fallback upstream ships). The OpenStructure image `images/ost.sif` is run through `bin/ost` (singularity).
* Python for the evaluator: the FoldForge env of `ff_2512` (numpy, pandas, biopython, networkx) with `parallelbar` on `PYTHONPATH` (`EVAL_ENV`, `PARALLELBAR` in `eval.sbatch`).
* CPU only. Never on the login node.

## Commands (after the F1 campaign; one evaluation job per run, ~4.5 h for 38,050 predictions on 32 CPUs)

```bash
cd ~/foldforge-scratch/paper-foldbench; export FOLDFORGE_REPO=$PWD PAPER_OUT=~/foldforge-scratch/paper; P=benchmarks/paper/F2_foldbench_accuracy
R=$PAPER_OUT/F1/h100
for spec in "af3_bf16_shard af3_bf16" "af3_fp8_shard af3_fp8"; do set -- $spec
  sbatch -J paper-F2-eval-$2 -o $PAPER_OUT/F2/h100/logs/%x_%j.log \
    --export=ALL,NAME=$2,PRED_DIRS=$R/$1/pred_s0:$R/$1/pred_s1,EVAL_OUT=$PAPER_OUT/F2/h100/eval/$2 $P/eval.sbatch     # multi-mode run: PRED_DIRS=$R/<run>/pred
done
# the kit run: lay its structures out like FoldForge's (CPU, seconds; symlinks <t>/<t>_seed{s}_sample{k}_pred.cif + ranking.csv from the kit's ranking_scores.csv, a big-mode target takes its out_big outputs):
$FFENV/bin/python benchmarks/paper/F1_foldbench_af3/kit/kit2pred.py af3 $R/af3kit $R/af3kit/pred      # then the same eval.sbatch with NAME=af3kit,PRED_DIRS=$R/af3kit/pred
# collect (CPU):
$FFENV/bin/python $P/collect.py --arm af3kit=$PAPER_OUT/F2/h100/eval/af3kit --arm af3_bf16=$PAPER_OUT/F2/h100/eval/af3_bf16 --arm af3_fp8=$PAPER_OUT/F2/h100/eval/af3_fp8 \
   --stamp af3_bf16=$R/af3_bf16_shard/stamp.json --stamp af3_fp8=$R/af3_fp8_shard/stamp.json --out $PAPER_OUT/F2/h100/results
```

`TOP1_ONLY=1` scores only the top-ranked sample of every target (~10x faster; top-1 only, `oracle` is NaN): that is how the earlier `foldbench_full_af3_accuracy.csv` was made.

## Output format

`f2_task_scores.csv`: `arm, task, metric, n, top1, oracle, foldforge_commit, engine_version, gpu, node, slurm_job_id, launch, precision` (percent; `n` = targets or interfaces after the NaN drop).
`f2_top1_wide.csv`, `f2_oracle_wide.csv`: rows = the nine tasks (`interface_antibody_antigen`, `interface_protein_protein`, `interface_protein_peptide`, `interface_protein_ligand`,
`interface_protein_dna`, `interface_protein_rna`, `monomer_protein`, `monomer_rna`, `monomer_dna`), columns = arms.
