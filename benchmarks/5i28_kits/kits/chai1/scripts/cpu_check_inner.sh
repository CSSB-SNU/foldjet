#!/bin/bash
# CPU-only (inside the cu130 container): build the Chai-1 inputs (prep.py) and load them with chai's own feature-context code (check.py).
H=$BENCH_DIR/chai1
source $H/scripts/env_in.sh
which kalign; kalign --version 2>&1 | head -1
cd $H/tmp
python -I $H/scripts/prep.py || exit 1
python $H/scripts/check.py ${CHECK_NS:-1 2}     # not -I: the shim on PYTHONPATH must load
