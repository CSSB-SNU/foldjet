#!/bin/bash
# Collect the WARM concurrency runs (run_concurrency_warm.sbatch): per run, the measurement starts when the 3rd target is done (the 3 warm-up
# targets of the list, one per worker for 3 workers, have captured their CUDA graphs); steady_s = last done - that moment, n = targets done
# after it, s_per_target = steady_s / n. Median over the repetitions, speed-up vs k1. awk only.
#   bash collect_concurrency_warm.sh [RUN_ROOT] [OUT_DIR]
set -euo pipefail
R=${1:-$HOME/foldforge-scratch/paper/F1/h100/concurrency_warm}; OUT=${2:-$(dirname "$0")/data}; mkdir -p "$OUT"
{
  echo "bin,cfg,rep,n_measured,steady_s,s_per_target,wall_s,done,failed,job"
  for d in "$R"/*/*_r*; do
    l=$d/pred/multi_events.log; j=$d/job_info.txt
    [ -f "$l" ] && grep -q "end rc=0" "$j" || continue
    bin=$(basename "$(dirname "$d")"); base=$(basename "$d"); cfg=${base%_r*}; rep=${base##*_r}
    awk -v bin="$bin" -v cfg="$cfg" -v rep="$rep" -v job="$(sed -n 's/^job=\([0-9]*\).*/\1/p' "$j")" -v w="$(sed -n 's/.*wall_s=\([0-9.]*\).*/\1/p' "$j")" '
      / done / { n++; t[n] = $4 + 0 }
      / summary / { for (i = 1; i <= NF; i++) { if ($i ~ /^done=/) dn = substr($i, 6); if ($i ~ /^failed=/) fl = substr($i, 8) } }
      END { if (n > 3) { s = t[n] - t[3]; printf "%s,%s,%s,%d,%.1f,%.3f,%s,%s,%s,%s\n", bin, cfg, rep, n - 3, s, s / (n - 3), w, dn, fl, job } }' "$l"
  done
} > "$OUT/concurrency_warm_runs.csv"
awk -F, 'NR > 1 { k = $1 "," $2; m[k]++; v[k, m[k]] = $6 }
  function med(k, c,   i, j, t, x) { for (i = 1; i <= c; i++) x[i] = v[k, i]
    for (i = 1; i <= c; i++) for (j = i + 1; j <= c; j++) if (x[j] + 0 < x[i] + 0) { t = x[i]; x[i] = x[j]; x[j] = t }
    return (c % 2) ? x[(c + 1) / 2] : (x[c / 2] + x[c / 2 + 1]) / 2 }
  END { for (k in m) md[k] = med(k, m[k]); print "bin,cfg,reps,s_per_target_median,speedup_vs_k1"
    for (k in m) { split(k, a, ","); b = a[1] ",k1"; printf "%s,%d,%.3f,%s\n", k, m[k], md[k], (b in md) ? sprintf("%.2f", md[b] / md[k]) : "" } }' \
  "$OUT/concurrency_warm_runs.csv" | { read -r h; echo "$h"; sort; } > "$OUT/concurrency_warm_median.csv"
echo "wrote $OUT/concurrency_warm_runs.csv ($(($(wc -l < "$OUT/concurrency_warm_runs.csv") - 1)) runs), $OUT/concurrency_warm_median.csv"
