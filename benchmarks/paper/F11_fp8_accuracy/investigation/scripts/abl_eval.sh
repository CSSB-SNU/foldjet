#!/bin/bash
# Submit the FoldBench evaluation (CPU) of one ablation config after its prediction job.
D=/home/psk6950/foldforge-scratch/paper/F11_investigation; R=/home/psk6950/foldforge-scratch/paper-306
cfg=$1; dep=$2
cd $R && sbatch --parsable -J F11ablev-$cfg ${dep:+-d afterok:$dep} -c 16 --mem=64G --time=03:00:00 -o $D/logs/%x_%j.log \
  --export=ALL,NAME=abl_$cfg,PRED_DIRS=$D/abl/$cfg/pred,EVAL_OUT=$D/abl/$cfg/eval,MW_OST_WORKERS=15,MW_DOCKQ_WORKERS=15,FFENV=$R/.pixi/envs/default \
  benchmarks/paper/F2_foldbench_accuracy/eval.sbatch
