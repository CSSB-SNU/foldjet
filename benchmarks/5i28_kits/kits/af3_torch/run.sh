#!/bin/bash
# usage: run.sh N      (N = 1 2 4 8 12 16) -- one 5I28 size of the AF3 (Torch) kit, mode fast, on the GPU the caller gives
# (CUDA_VISIBLE_DEVICES set by the caller, on a GPU compute node -- never the login node). Full log: logs/x<N>.log.
# Last stdout line:
#   KIT af3_torch n=<N> exit=<rc> fwd=[<s>,...] warm_median=<s> templates_real=<k>/4 msa_rows=<r>
# fwd    = the kit's "[af3-torch-opt] ITEM ... forward_s=X" per seed, in log order (first = cold): torch.cuda.synchronize,
#          A.inference() = trunk x4 + 5-sample sampler + confidence + distogram, torch.cuda.synchronize
#          (docs/records/anthropic_kits_fwd_definition_audit_20261029.md); featurise/load/write are outside it.
# warm_median = median of fwd[1:].
# templates_real = min( min over ITEM lines of 4*a/b from "templates=a/b" (a/b = real/declared slots x chains, b = 4n),
#                       tmpl_distinct_min of the FEATPROBE line,
#                       4*evaluated/slots of the kit's template_dedupe LEVER line when it shows evaluated < slots )
#                  (0 if the ITEM or FEATPROBE evidence is missing)
# msa_rows = msa_rows of the FEATPROBE line (kit's alphafold3 featuriser env on the same JSON, resolve_msa_overlaps=False as the kit).
set -u
N=${1:?usage: run.sh N}
D=$BENCH_DIR/af3_torch
K=$KITS_DIR
LOG=$D/logs/x$N.log
mkdir -p $D/logs
if [ ! -f $D/inputs/x$N/5i28.json ]; then
  echo "KIT af3_torch n=$N exit=2 fwd=[] warm_median=NA templates_real=0/4 msa_rows=0"; exit 2
fi
# Warm JIT caches = a private copy of the kit's prebuilt jit/ (inductor + triton, ~1 GB) and its torch kernel cache; the kit tree
# is only read. Once; locked for parallel n.
(
  flock 9
  if [ ! -e $D/jit/.copied ]; then
    rm -rf $D/jit.part $D/jit && mkdir -p $D/jit.part && cp -a $K/af3_torch/jit/. $D/jit.part/ \
      && chmod -R u+w $D/jit.part \
      && find $D/jit.part -name '__grp__*.json' -exec sed -i "s#$K/af3_torch/jit/#$D/jit/#g" {} + \
      && mkdir -p $D/cache && cp -a $K/af3_torch/cache/torch $D/cache/ \
      && mv $D/jit.part $D/jit && touch $D/jit/.copied
  fi
) 9>$D/.jit.lock
export CU130_EXTRA_LD=$K/libstdcxx:$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib
rm -f $D/out/x$N/rc   # no stale rc if the container fails to start
{
  echo "RUN af3_torch n=$N host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} jit_copied=$([ -e $D/jit/.copied ] && echo yes || echo NO)"
  $K/cu130run.sh bash $D/inner.sh $N
  echo "CONTAINER_RC=$?"
} > $LOG 2>&1
RC=$(cat $D/out/x$N/rc 2>/dev/null || sed -n 's/^CONTAINER_RC=//p' $LOG | tail -1)
FWD=$(grep -E '^\[af3-torch-opt\] ITEM ' $LOG | grep -oE 'forward_s=[0-9.]+' | cut -d= -f2 | paste -sd, -)
WARM=$(echo "$FWD" | tr , '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END{if(NR==0){print "NA"} else if(NR%2){printf "%.3f\n",a[(NR+1)/2]} else {printf "%.3f\n",(a[NR/2]+a[NR/2+1])/2}}')
# ITEM "templates=a/b" counts template slots x chains (b = 4n): per chain 4a/b; an ITEM line without it counts 0
KLOG=$(grep -E '^\[af3-torch-opt\] ITEM ' $LOG | awk '{k=0; if (match($0, / templates=[0-9]+\/[0-9]+ /)) {split(substr($0, RSTART+11, RLENGTH-12), t, "/"); if (t[2] > 0 && t[2] % 4 == 0) k = int(4 * t[1] / t[2])} print k}' | sort -n | head -1)
KPRB=$(grep -oE 'FEATPROBE .*tmpl_distinct_min=[0-9]+' $LOG | sed -E 's#.*tmpl_distinct_min=##' | tail -1)
DEDUP=$(grep -E 'LEVER name=template_dedupe state=on' $LOG | grep -oE 'evaluated=[0-9]+ scans=[0-9]+ slots=[0-9]+' | tail -1 \
        | awk '{split($1,e,"=");split($3,s,"="); if(s[2]>0 && e[2]<s[2]) print int(4*e[2]/s[2]); else print 4}')
MSA=$(grep -oE 'FEATPROBE .* msa_rows=[0-9]+' $LOG | sed -E 's#.* msa_rows=##' | tail -1)
TR=0
if [ -n "$KLOG" ] && [ -n "$KPRB" ]; then
  TR=$(( KLOG < KPRB ? KLOG : KPRB )); [ -n "$DEDUP" ] && TR=$(( TR < DEDUP ? TR : DEDUP ))
fi
echo "KIT af3_torch n=$N exit=${RC:-99} fwd=[${FWD}] warm_median=${WARM} templates_real=${TR}/4 msa_rows=${MSA:-0}" | tee -a $LOG
