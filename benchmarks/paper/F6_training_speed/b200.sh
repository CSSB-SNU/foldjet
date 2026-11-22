#!/bin/bash
# F6 on the B200 (host b200, GPUs 0-3, no Slurm). One lane = one GPU (CUDA_VISIBLE_DEVICES from the lane driver). run_f6.sh does the matrix of MODELS x ARMS x REPEATS;
# the campaign calls it once per (model, arm) (3 repeats). The 192 GB card: ladder for the non-engine arms "0 24 12 8" (first value that fits is used; recorded).
#   MODELS=af3 ARMS=miniworld [SMOKE=1] benchmarks/paper/F6_training_speed/b200.sh
export CARD=b200
export AUG_LADDER_OTHER=${AUG_LADDER_OTHER:-"0 24 12 8 6 4"}
exec "$(dirname "$(readlink -f "$0")")/run_f6.sh"
