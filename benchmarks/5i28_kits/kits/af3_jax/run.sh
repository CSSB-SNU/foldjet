#!/bin/bash
# usage: run.sh N      (N = 1 2 4 8 12 16) -- one 5I28 size of the AF3 (JAX) kit, mode fast, on the GPU the caller gives
# (CUDA_VISIBLE_DEVICES set by the caller, on a GPU compute node -- never the login node). Full log: logs/x<N>.log.
# Last stdout line:
#   KIT af3_jax n=<N> exit=<rc> fwd=[<s>,...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
# fwd    = the kit's "Running model inference with seed S took X seconds" per seed, in log order (first = cold). This is the
#          jitted model call (trunk x4 + 5-sample sampler + confidence) + the device-to-host copy of its outputs
#          (docs/records/anthropic_kits_fwd_definition_audit_20261029.md); featurisation and output writing are outside it.
# warm_median = median of fwd[1:].
# templates_real = min( min over the kit's "[af3-jax-opt] TEMPLATES chain=X declared=4 kept=4 real=K/4" lines,
#                       tmpl_distinct_min of the FEATPROBE line )   (0 if either is missing)
# msa_rows = msa_rows of the FEATPROBE line (kit's own alphafold3 featuriser on the same JSON, resolve_msa_overlaps as the kit).
set -u
N=${1:?usage: run.sh N}
D=$BENCH_DIR/af3_jax
K=$KITS_DIR
LOG=$D/logs/x$N.log
mkdir -p $D/logs
if [ ! -f $D/inputs/x$N/5i28.json ]; then
  echo "KIT af3_jax n=$N exit=2 fwd=[] warm_median=NA templates_real=0/4 msa_rows=0"; exit 2
fi
# Warm compile cache = a private copy of the kit's prebuilt jitroot (the kit tree is only read). Once; locked for parallel n.
(
  flock 9
  if [ ! -e $D/jitroot/.copied ]; then
    rm -rf $D/jitroot.part $D/jitroot && mkdir -p $D/jitroot.part && cp -a $K/af3_jax/jitroot/. $D/jitroot.part/ \
      && chmod -R u+w $D/jitroot.part \
      && find $D/jitroot.part -name '__grp__*.json' -exec sed -i "s#$K/af3_jax/jitroot/#$D/jitroot/#g" {} + \
      && mv $D/jitroot.part $D/jitroot && touch $D/jitroot/.copied
  fi
) 9>$D/.jitroot.lock
rm -f $D/out/x$N/rc   # no stale rc if the container fails to start
{
  echo "RUN af3_jax n=$N host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} jitroot_copied=$([ -e $D/jitroot/.copied ] && echo yes || echo NO)"
  $K/cu130run.sh bash $D/inner.sh $N
  echo "CONTAINER_RC=$?"
} > $LOG 2>&1
RC=$(cat $D/out/x$N/rc 2>/dev/null || sed -n 's/^CONTAINER_RC=//p' $LOG | tail -1)
FWD=$(grep -oE 'Running model inference with seed [0-9]+ took [0-9.]+ seconds' $LOG | awk '{print $(NF-1)}' | paste -sd, -)
WARM=$(echo "$FWD" | tr , '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END{if(NR==0){print "NA"} else if(NR%2){printf "%.3f\n",a[(NR+1)/2]} else {printf "%.3f\n",(a[NR/2]+a[NR/2+1])/2}}')
KLOG=$(grep -oE '\[af3-jax-opt\] TEMPLATES chain=[A-Za-z]+ declared=4 kept=[0-9]+ real=[0-9]+/4' $LOG | sed -E 's#.*real=([0-9]+)/4#\1#' | sort -n | head -1)
KPRB=$(grep -oE 'FEATPROBE .*tmpl_distinct_min=[0-9]+' $LOG | sed -E 's#.*tmpl_distinct_min=##' | tail -1)
MSA=$(grep -oE 'FEATPROBE .* msa_rows=[0-9]+' $LOG | sed -E 's#.* msa_rows=##' | tail -1)
TR=0; [ -n "$KLOG" ] && [ -n "$KPRB" ] && TR=$(( KLOG < KPRB ? KLOG : KPRB ))
echo "KIT af3_jax n=$N exit=${RC:-99} fwd=[${FWD}] warm_median=${WARM} templates_real=${TR}/4 msa_rows=${MSA:-0}" | tee -a $LOG
