#!/bin/bash
# Collect the kit-side 5I28 results of a benchmark directory into a CSV and a raw per-forward JSON.
# usage: BENCH_DIR=<dir> collect_kit_results.sh OUT_PREFIX     -> OUT_PREFIX.csv, OUT_PREFIX.json
# Reads the done markers (<BENCH_DIR>/done_final/kit_<kit>_x<n> and done_sep/sep_<kit>_x<n>, written by queue3.sh / run_multi.sh).
# A marker's fwd=[...] list is every forward of the cell in log order (first = cold). A list that is its own half repeated twice
# (Boltz-2: the kit prints each PHASE line to stdout and stderr) is reduced to one half. Warm = every forward after the first;
# warm_median of an even count is the mean of the middle two. Status: marker exit=0 -> ok; Chai-1 exit=1 with forwards -> ok (the kit
# exits 1 after its forward-only span); empty list with a CUDA OOM in the cell log -> oom; otherwise failed.
set -u
B=${BENCH_DIR:?set BENCH_DIR}; OUT=${1:?usage: collect_kit_results.sh OUT_PREFIX}
echo "model,n,tokens,warm_min_s,warm_median_s,warm_max_s,n_forwards,n_warm,status,cold_s,job_id,node,source_log,note" > $OUT.csv
echo "[" > $OUT.json; first=1
for kit in af3_jax af3_torch openfold3 rf3 chai1 esmfold2 boltz2 protenix_v1 protenix_v2 opendde; do
  for n in 1 2 4 8 12 16; do
    case $kit in protenix_v1|protenix_v2|opendde) m=$B/done_sep/sep_${kit}_x$n; src=$kit/logs/x$n.log;;
                 boltz2|esmfold2) m=$B/done_final/kit_${kit}_x$n; src=$kit/logs/all.log;;
                 *) m=$B/done_final/kit_${kit}_x$n; src=$kit/logs/x$n.log;; esac
    [ -s $m ] || { echo "$mname,$n,$((128*n)),,,,0,0,missing,,,,$src," >> $OUT.csv; continue; }
    line=$(cat $m); mname=$kit; [ $kit = rf3 ] && mname=rosettafold3
    job=$(echo "$line" | grep -oE 'job=[0-9]+' | cut -d= -f2); node=$(echo "$line" | grep -oE 'host=[a-z0-9]+' | cut -d= -f2)
    ex=$(echo "$line" | grep -oE 'exit=[0-9a-z]+' | head -1 | cut -d= -f2); st=$(echo "$line" | grep -oE 'status=[a-z]+' | cut -d= -f2)
    fwd=$(echo "$line" | sed -E 's/.*fwd=\[([^]]*)\].*/\1/' | tr ',' ' ')
    set -- $fwd; c=$#
    if [ $((c % 2)) -eq 0 ] && [ $c -gt 0 ]; then h=$((c/2)); [ "${*:1:$h}" = "${*:$((h+1)):$h}" ] && fwd="${*:1:$h}"; fi
    status=failed; note=""
    if [ -n "$fwd" ]; then status=ok; [ "$ex" = "1" ] && note="kit exit=1 after its forward-only span"; fi
    [ "${st:-}" = oom ] && status=oom
    if [ -z "$fwd" ] && [ "$status" = failed ] && [ -f $B/$src ] && grep -aqiE "CUDA out of memory|OutOfMemory" $B/$src; then status=oom; note="CUDA OOM in $src"; fi
    echo "$fwd" | awk -v kit=$mname -v n=$n -v st=$status -v job=$job -v node=$node -v src=$src -v note="$note" -v json=$OUT.json -v first=$first '
      { nf=split($0,a," "); cold=(nf>0)?a[1]:""; nw=nf>1?nf-1:0
        for(i=2;i<=nf;i++) w[i-1]=a[i]+0
        for(i=1;i<=nw;i++) for(j=i+1;j<=nw;j++) if(w[j]<w[i]){t=w[i];w[i]=w[j];w[j]=t}
        if(nw){ med=(nw%2)?w[(nw+1)/2]:(w[nw/2]+w[nw/2+1])/2; mn=w[1]; mx=w[nw] }
        raw=""; for(i=1;i<=nf;i++) raw=raw (i>1?",":"") a[i]
        printf "%s,%d,%d,%s,%s,%s,%d,%d,%s,%s,%s,%s,%s,%s\n", kit,n,128*n,(nw?sprintf("%.3f",mn):""),(nw?sprintf("%.3f",med):""),(nw?sprintf("%.3f",mx):""),nf,nw,st,cold,job,node,src,note >> (json ".csvrow")
        printf "%s{\"model\":\"%s\",\"n\":%d,\"tokens\":%d,\"status\":\"%s\",\"job_id\":\"%s\",\"node\":\"%s\",\"source_log\":\"%s\",\"forwards_s\":[%s],\"note\":\"%s\"}", (first?"":",\n"), kit,n,128*n,st,job,node,src,raw,note >> json }'
    first=0; cat $OUT.json.csvrow >> $OUT.csv; rm -f $OUT.json.csvrow
  done
done
echo "]" >> $OUT.json
