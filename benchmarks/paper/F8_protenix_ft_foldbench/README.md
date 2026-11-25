# F8: the fine-tuned Protenix v1 (10 epochs) on the whole FoldBench, against the public Protenix v1

**What is measured.** Protenix v1 fine-tuned for 10 epochs (PDB-only fine-tune with the engine, `ft_engine_fp32d`: protenix recipe, fp32 diffusion head, antibody-protein /
protein-RNA sampler weights 10 / 20; checkpoint `epoch0010.miniworld.pt`) against the released Protenix v1 weights, both on all 1,522 FoldBench targets through
FoldJet (5 trunk seeds x 5 samples, 200 diffusion steps, the model's default trunk passes, bf16, compile + CUDA graphs, the same `fb_multi.py` as F1), scored with
**F2's rules** (FoldBench evaluator, upstream `task_score_summary.py` selection): top-1 (ranking score) and oracle (best of 25) per task.

Possible train / test overlap: the fine-tune data is PDB-only up to its cut-off; the FoldBench targets are post-2022 PDB entries, but the overlap with the fine-tune
set (BioMolDB_20261014) has not been audited. Report the fine-tune result with that caveat (it is why the antibody-antigen gain of the 113-target subset was not used before).

## Files

| file | what |
|---|---|
| `run_f8.sh` | submits both runs (`released`, `ft10`) through F1's `foldjet/run_ff.sbatch` (`EXP=F8 MODEL=protenix1 CKPT=...`, default `LAUNCH=multi`: accuracy only, the wall time is not a result) and, behind each, the CPU evaluation (`F2_foldbench_accuracy/eval.sbatch`, all 25 samples) |
| `checkpoints.txt` | the two arms and the checkpoint path (on the H100 cluster, not in the repository) |
| `collect.py` | the two evaluation dirs -> `f8_task_scores.csv`, `f8_top1.csv` (`,released,ft10,diff`: the shape of the paper's `tables/finetune_foldbench.csv`), `f8_oracle.csv` |
| `results/` | the CSVs of the campaign |

Run outputs: `$PAPER_OUT/F8/h100/protenix1_{released,ft10}/` (`pred/`, `stamp.json` with FoldJet commit / engine version / GPU / node / job id and the checkpoint path, `launch.txt`),
evaluations `$PAPER_OUT/F8/h100/eval/protenix1_{released,ft10}/`.

## Full-campaign commands (after the engine release; same FoldJet commit and engine as F1)

```bash
cd ~/foldforge-scratch/paper-foldbench
export FOLDFORGE_REPO=$PWD FFENV=<FoldJet env with the released engine> PAPER_OUT=~/foldforge-scratch/paper
bash benchmarks/paper/F8_protenix_ft_foldbench/run_f8.sh          # 2 x (2-GPU run job ~4.5 h wall, then a ~4.5 h CPU evaluation)
# when both evaluations are done (CPU, FoldJet env python):
$FFENV/bin/python benchmarks/paper/F8_protenix_ft_foldbench/collect.py \
   --released $PAPER_OUT/F8/h100/eval/protenix1_released --ft10 $PAPER_OUT/F8/h100/eval/protenix1_ft10 \
   --stamp-released $PAPER_OUT/F8/h100/protenix1_released/stamp.json --stamp-ft10 $PAPER_OUT/F8/h100/protenix1_ft10/stamp.json \
   --out $PAPER_OUT/F8/h100/results
cp $PAPER_OUT/F8/h100/results/f8_*.csv benchmarks/paper/F8_protenix_ft_foldbench/results/
```

Expected cost (earlier records): Protenix v1 released, whole FoldBench, 2 H100 in `fb_multi`: 4 h 31 min (engine 3.0.1, job 58553); the fine-tune ~4.5 h (engine 2.5.14, job 57793);
the evaluation of 38,050 predictions ~4.5 h on 32 CPUs. Earlier numbers (engine 2.5.14, top-1 only): `tables/finetune_foldbench.csv` of the paper tree.
A walltime-style run is ONE continuous job; if a submit is refused, stop and report.
