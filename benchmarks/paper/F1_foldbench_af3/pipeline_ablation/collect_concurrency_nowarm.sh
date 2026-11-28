#!/bin/bash
# F1 pipeline addendum (2026-11-27): collect the concurrency x seed-grouping runs (run_concurrency.sbatch) into
#   concurrency_nowarm_runs.csv    one row per run: bin, cfg, rep, wall_s (job, model loads included), proc_s (first worker ready -> last target
#                           finished, model loads excluded), done, failed, job, node
#   concurrency_nowarm_median.csv  per bin x cfg: median proc_s / wall_s over the repetitions, speed-up vs k1 (median proc_s of k1 / of cfg)
# awk only (no python): runs on the login node.   bash collect_concurrency.sh RUN_ROOT OUT_DIR
set -euo pipefail
R=${1:-$HOME/foldforge-scratch/paper/F1/h100/concurrency}; OUT=${2:-$(dirname "$0")/data}; mkdir -p "$OUT"
{
  echo "bin,cfg,rep,wall_s,proc_s,done,failed,job,node"
  for d in "$R"/*/*_r*; do
    l=$d/pred/multi_events.log; j=$d/job_info.txt
    [ -f "$l" ] && grep -q "end rc=0" "$j" || continue
    bin=$(basename "$(dirname "$d")"); base=$(basename "$d"); cfg=${base%_r*}; rep=${base##*_r}
    awk -v bin="$bin" -v cfg="$cfg" -v rep="$rep" \
        -v job="$(sed -n 's/^job=\([0-9]*\).*/\1/p' "$j")" -v node="$(sed -n 's/.* host=\([^ ]*\).*/\1/p' "$j")" '
      / gpu .*first_ready_s/ { for (i = 1; i <= NF; i++) {
          if ($i ~ /^first_ready_s=/) { v = substr($i, 15) + 0; if (fr == "" || v < fr) fr = v }
          if ($i ~ /^last_finish_s=/) { v = substr($i, 15) + 0; if (v > lf) lf = v } } }
      / summary / { for (i = 1; i <= NF; i++) { if ($i ~ /^wall_s=/) w = substr($i, 8); if ($i ~ /^done=/) dn = substr($i, 6); if ($i ~ /^failed=/) fl = substr($i, 8) } }
      END { if (lf) printf "%s,%s,%s,%s,%.1f,%s,%s,%s,%s\n", bin, cfg, rep, w, lf - fr, dn, fl, job, node }' "$l"
  done
} > "$OUT/concurrency_nowarm_runs.csv"
awk -F, 'NR > 1 { k = $1 "," $2; n[k]++; p[k, n[k]] = $5; w[k, n[k]] = $4 }
  function med(a, k, m,   i, j, t, x) { for (i = 1; i <= m; i++) x[i] = a[k, i]
    for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (x[j] + 0 < x[i] + 0) { t = x[i]; x[i] = x[j]; x[j] = t }
    return (m % 2) ? x[(m + 1) / 2] : (x[m / 2] + x[m / 2 + 1]) / 2 }
  END { for (k in n) { mp[k] = med(p, k, n[k]); mw[k] = med(w, k, n[k]) }
    print "bin,cfg,reps,proc_s_median,wall_s_median,speedup_vs_k1_proc"
    for (k in n) { split(k, a, ","); b = a[1] ",k1"; printf "%s,%d,%.1f,%.1f,%s\n", k, n[k], mp[k], mw[k], (b in mp) ? sprintf("%.2f", mp[b] / mp[k]) : "" } }' \
  "$OUT/concurrency_nowarm_runs.csv" | { read -r h; echo "$h"; sort; } > "$OUT/concurrency_nowarm_median.csv"
echo "wrote $OUT/concurrency_nowarm_runs.csv ($(($(wc -l < "$OUT/concurrency_nowarm_runs.csv") - 1)) runs), $OUT/concurrency_nowarm_median.csv"
