#!/bin/bash
# The DDP scaling matrix on vast5090 (nr_nt pf16 = the v2.4.0 phase-1a production config, L384, 1 sample per GPU). Sequential; skips done tags.
D=/home/psk6950/mwe/agents/mw_ddp; cd $D
run() { tag=$1; shift; [ -s results/$tag/summary.json ] && { echo "skip $tag"; return; }; "$@" > logs/$tag.log 2>&1; grep -a "^Step\|^\[mw_ddp\]\|^===\|Error\|error:" logs/$tag.log | grep -v "Step  *[0-9]* .*Step" | tail -14; }
export STEPS=${STEPS:-25}
for ga in 1 4 16; do
  run n1g1_ga$ga   ./run_one.sh n1g1_ga$ga   1       train.grad_accum_steps=$ga
  run n1g0_ga$ga   ./run_one.sh n1g0_ga$ga   0       train.grad_accum_steps=$ga
  run n2g12_ga$ga  ./run_one.sh n2g12_ga$ga  1,2     train.grad_accum_steps=$ga
  run n4_ga$ga     ./run_one.sh n4_ga$ga     0,1,2,3 train.grad_accum_steps=$ga
done
run n4_ga2      ./run_one.sh n4_ga2      0,1,2,3 train.grad_accum_steps=2
run n2g01_ga1   ./run_one.sh n2g01_ga1   0,1     train.grad_accum_steps=1
PIN=0 run n4_ga1_nopin ./run_one.sh n4_ga1_nopin 0,1,2,3 train.grad_accum_steps=1
# loss sanity: short warmup so the lr is real within the run
STEPS=60 VALIDATE=1 run n4_ga4_loss ./run_one.sh n4_ga4_loss 0,1,2,3 train.grad_accum_steps=4 train.warmup_steps=5
echo CAMPAIGN-END $(date -Is)
