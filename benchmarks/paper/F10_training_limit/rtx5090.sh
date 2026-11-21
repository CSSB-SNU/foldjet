#!/bin/bash
# F10 on the RTX 5090 box (cssb-rtx), one shared GPU: ~/mwe/gpu.sh locks per CALL, so every TRIAL (one process) takes the lock itself.
#   benchmarks/paper/F10_training_limit/rtx5090.sh      (SMOKE=1 for the plumbing check; resumable)
export CARD=rtx5090
export TRIAL_WRAPPER="$HOME/mwe/gpu.sh 2600"
exec "$(dirname "$(readlink -f "$0")")/run_f10.sh"
