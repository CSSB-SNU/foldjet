#!/bin/bash
# CPU-only (inside the cu130 container): Boltz-2's own parse of the bench YAMLs (templates + MSA).
H=$BENCH_DIR/boltz2
source $H/scripts/env_in.sh
cd $H/tmp
python -I $H/scripts/check.py ${CHECK_NS:-1 2 16}
