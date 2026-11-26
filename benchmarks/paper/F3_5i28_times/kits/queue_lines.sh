#!/bin/bash
# Print the queue lines ("name|command") of one Anthropic kit for `queue3.sh` (the 2026-11-18 steady protocol: forward 1 = cold, forward 2 dropped, then
# 5 timed; mode fast, a cell that runs out of memory under fast -> the kit's big mode in a fresh process: fb.sh).
#   KIT_BENCH=<harness dir> DONE=<done dir> bash queue_lines.sh KIT [sizes...]
# KIT: af3_jax af3_torch openfold3 openfold3_p2 rf3 chai1 esmfold2 esmfold2_fast boltz2 protenix_v1 protenix_v2 opendde intellifold2 intellifold2_notxla
set -u
B=${KIT_BENCH:?KIT_BENCH}; HERE=$(cd "$(dirname "$0")" && pwd); KIT=${1:?kit}; shift; SIZES=${*:-1 2 4 8 12 16}
FB="bash $B/fb.sh"; p() { echo "$1|$2"; }
for n in $SIZES; do
  case $KIT in
    af3_jax)   p kit_af3_jax_x$n "$FB 7 $B/af3_jax/logs/x$n.log bash $B/af3_jax/run.sh $n";;
    af3_torch) p kit_af3_torch_x$n "$FB 7 $B/af3_torch/logs/x$n.log bash $B/af3_torch/run.sh $n";;
    openfold3) p kit_openfold3_x$n "COPIES=7 $FB 7 $B/openfold3/logs/x$n.log bash $B/openfold3/run.sh $n";;
    openfold3_p2) p kit_openfold3_p2_x$n "COPIES=7 KIT_CONFIG=${KIT_CONFIG:-h100} $FB 7 $B/openfold3_p2/logs/x$n.log bash $B/openfold3_p2/run.sh $n";;   # Anthropic openfold3 kit: OpenFold3 0.4.1 + preview-2 weights
    rf3)       p kit_rf3_x$n "COPIES=7 $FB 7 $B/rf3/logs/x$n.log bash $B/rf3/run.sh $n";;
    chai1)     if [ "$n" = 16 ]; then p kit_chai1_x16 "FWDS=6 $FB 6 $B/chai1/logs/x16.log bash $B/chai1/run.sh 16"   # one fold ~8 min: 6 forwards (4 timed)
               else p kit_chai1_x$n "FWDS=7 $FB 7 $B/chai1/logs/x$n.log bash $B/chai1/run.sh $n"; fi;;
    protenix_v1|protenix_v2|opendde) p sep_${KIT}_x$n "$FB 7 $B/$KIT/logs/x$n.log bash $B/protenix_family_run.sh $KIT $n";;
    intellifold2) p kit_intellifold2_x$n "$FB 7 $B/intellifold2/logs/kit_x$n.log bash $B/intellifold2/run.sh $n";;
    intellifold2_notxla) p kit_intellifold2_notxla_x$n "KITTAG=notxla MODEL_OPT_LEVERS_OFF=TRIATT_XLA $FB 7 $B/intellifold2/logs/kit_notxla_x$n.log bash $B/intellifold2/run.sh $n";;
    boltz2|esmfold2|esmfold2_fast) p kit_${KIT}_all "bash $HERE/run_all.sh $KIT"; break;;
    *) echo "unknown kit $KIT" >&2; exit 2;;
  esac
done
