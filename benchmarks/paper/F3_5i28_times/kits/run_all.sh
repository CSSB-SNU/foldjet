#!/bin/bash
# Kit side, kits that run all sizes in ONE process (Boltz-2, ESMFold2, ESMFold2-Fast): mode fast for every n, then every n that ran out of memory under
# fast again in a fresh process under the kit's own `big` mode (the only retry the protocol allows). Same logic as the 2026-11-18 harness scripts
# (`run_all.sh` of the kit directories) with the done directory and the harness path taken from the environment.
#   KIT_BENCH=<harness dir> DONE=<done dir> bash run_all.sh boltz2|esmfold2|esmfold2_fast
set -u
KIT=${1:?kit}; B=${KIT_BENCH:?KIT_BENCH}; DONE=${DONE:?DONE}; mkdir -p "$DONE"; export DONE
case $KIT in
  boltz2)        RUN="bash $B/boltz2/run_multi.sh";        NAME=boltz2;        BAD='status=ok';  NEG=1;;
  esmfold2)      RUN="bash $B/esmfold2/run_multi.sh";      NAME=esmfold2;      BAD='status=oom'; NEG=0;;
  esmfold2_fast) RUN="env VARIANT=fast ESM_REPO=biohub/ESMFold2-Fast KNAME=esmfold2_fast bash $B/esmfold2_fast/run_multi.sh"; NAME=esmfold2_fast; BAD='status=oom'; NEG=0;;
  *) echo "unknown kit $KIT"; exit 2;;
esac
MODE=fast TAG=all $RUN
for n in 1 2 4 8 12 16; do
  M=$DONE/kit_${NAME}_x$n
  if [ "$NEG" = 1 ]; then grep -q "$BAD" "$M" 2>/dev/null && continue; else grep -q "$BAD" "$M" 2>/dev/null || continue; fi
  echo "FAST not ok n=$n -> big, fresh process"
  if [ "$KIT" = boltz2 ]; then MODE=big TAG=big$n SIZES="$n:7" $RUN; else MODE=big TAG=big$n SIZES=$n $RUN; fi
done
