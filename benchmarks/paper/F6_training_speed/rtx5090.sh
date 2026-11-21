#!/bin/bash
# F6 on the RTX 5090 box (cssb-rtx): ONE GPU, shared by every agent -> everything runs under ~/mwe/gpu.sh (flock, 16 GB RAM cap).
#   ~/mwe/gpu.sh 20000 benchmarks/paper/F6_training_speed/rtx5090.sh      (SMOKE=1 for the plumbing check)
# The 32 GB card needs a longer augmentation ladder for the non-engine arms; the default here is "0 24 12 8 6 4".
export CARD=rtx5090
export AUG_LADDER_OTHER=${AUG_LADDER_OTHER:-"0 24 12 8 6 4"}
exec "$(dirname "$(readlink -f "$0")")/run_f6.sh"
