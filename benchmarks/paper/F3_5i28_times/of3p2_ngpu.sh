#!/bin/bash
# F3 addendum (2026-11-27): OpenFold3 preview 2 on 5I28, both sides, in ONE job over NG GPUs of one node (a work queue: one task per GPU at a
# time, each task pinned to its own CPUS_PER_TASK cores). Tasks:
#   FoldJet  openfold3-preview2  arm engine (and engine_fp8 when FP8=1)  repetitions 1-3   (run_ff.py, all sizes in one process, as h100.sbatch)
#   kit      openfold3_p2 (Anthropic openfold3 kit = OpenFold3 0.4.1 + preview-2 weights)  repetitions 1-3, one fresh process per size
#            (queue_lines.sh -> fb.sh: fast, big in a fresh process after a fast OOM), done markers as queue3.sh writes them
# Resumable: finished FoldJet cells (cells.jsonl) and kit sizes (done_final/<name>) are skipped.
#   CARD=h100 NG=4 FP8=1 sbatch ... of3p2_ngpu.sh        (the caller's sbatch line sets partition, GPUs, CPUs, memory, node)
set -uo pipefail
HERE=${FF_REPO:?FF_REPO}/benchmarks/paper
source "$HERE/common/cards/${CARD:?CARD}.env"
NG=${NG:-4}; CPT=${CPUS_PER_TASK:-8}; M=openfold3-preview2
Q=$PAPER_OUT/F3/$CARD/of3p2_queue_${SLURM_JOB_ID}; mkdir -p "$Q"
ALLOWED=$(taskset -pc $$ | cut -d: -f2 | tr ',' '\n' | awk -F- '{if (NF==2) for(i=$1;i<=$2;i++) print i; else print $1}' | tr -d ' ' | paste -sd' ')
echo "JOB $SLURM_JOB_ID host=$(hostname) card=$CARD NG=$NG cpus/task=$CPT allowed=[$ALLOWED] $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,driver_version --format=csv,noheader
{
  # FoldJet side: not needed by default (FoldJet runs openfold3 and openfold3-preview2 through the same AlphaFold3 class and settings; only
  # the weights differ, so the existing openfold3 rows are its times; n=1 check 2026-11-27: 0.403 s vs 0.39 s). FF=1 adds it.
  ARMS=""; [ "${FF:-0}" = 1 ] && ARMS="engine"; [ "${FF:-0}" = 1 ] && [ "${FP8:-0}" = 1 ] && ARMS="engine engine_fp8"
  for rep in 1 2 3; do
    # the kit's large sizes first in the list so they do not all land at the end
    for line in $(KIT_BENCH=$KIT_BENCH DONE=x bash "$HERE/F3_5i28_times/kits/queue_lines.sh" openfold3_p2 16 12 8 4 2 1 | tr ' ' '\037'); do
      echo "kit|$rep|${line//$'\037'/ }"
    done
    for arm in $ARMS; do echo "ff|$rep|$arm"; done
  done
} > "$Q/tasks.txt"
cat "$Q/tasks.txt"
run() {
  IFS='|' read -r kind rep rest <<< "$1"
  while :; do for s in $(seq 0 $((NG - 1))); do exec 9>"$Q/slot$s"; if flock -n 9; then
    CPUS=$(echo $ALLOWED | tr ' ' '\n' | sed -n "$((s * CPT + 1)),$((s * CPT + CPT))p" | paste -sd, -)
    export CUDA_VISIBLE_DEVICES=$s OMP_NUM_THREADS=$CPT MKL_NUM_THREADS=$CPT
    t0=$(date +%s)
    if [ "$kind" = ff ]; then
      OUT=$PAPER_OUT/F3/$CARD/${rest}_rep$rep/$M; mkdir -p "$OUT"
      taskset -c "$CPUS" python -u "$HERE/F3_5i28_times/run_ff.py" --model $M --arm "$rest" --out "$OUT" --rep "$rep" --repeats 6 \
        --sizes 1 2 4 8 12 16 > "$OUT/run_${SLURM_JOB_ID}.log" 2>&1
      res="rc=$? $(grep -c '^CELL' "$OUT/run_${SLURM_JOB_ID}.log") cells: $(grep -oE '^CELL [^ ]+ [^ ]+ n=[0-9]+ [a-z]+ steady=[0-9.]{1,6}' "$OUT/run_${SLURM_JOB_ID}.log" | cut -d' ' -f4- | paste -sd';')"
    else
      IFS='|' read -r name cmd <<< "$rest"
      D=$PAPER_OUT/F3/$CARD/kits_rep$rep/done_final; mkdir -p "$D"
      n=${name##*_x}; cmd=${cmd//logs\/x$n.log/logs\/x${n}_r$rep.log}   # this repetition's own kit log (fb.sh reads it for the OOM check)
      if [ -s "$D/$name" ]; then res="SKIP done: $(cat "$D/$name")"; else
        out=$(RUNTAG=_r$rep taskset -c "$CPUS" bash -c "$cmd" 2>&1 | tail -1)
        case "$out" in KIT*exit=0*) echo "job=$SLURM_JOB_ID host=$(hostname) gpu=$s cpus=$CPUS $out" > "$D/$name";; esac
        cp "$KIT_BENCH/openfold3_p2/logs/x${n}_r$rep.log" "$D/$name.kitlog" 2>/dev/null
        cp "$KIT_BENCH/openfold3_p2/logs/x${n}_big_r$rep.log" "$D/$name.big.kitlog" 2>/dev/null
        res="$out"
      fi
    fi
    echo "$(date +%T) gpu$s $kind rep$rep ${rest%%|*} wall=$(( $(date +%s) - t0 ))s :: $res"
    flock -u 9; return; fi; done; sleep 3; done
}
export -f run; export Q NG CPT M HERE ALLOWED PAPER_OUT CARD KIT_BENCH SLURM_JOB_ID
# preflight + compile-cache warm-up, alone on GPU 0 (a discarded n=1 kit run: concurrent first runs would race on the kit's JIT cache)
CUDA_VISIBLE_DEVICES=0 RUNTAG=_warm COPIES=3 bash "$KIT_BENCH/openfold3_p2/run.sh" 1 | tee "$Q/warm.txt"
grep -q 'exit=0 .*templates_real=4/4' "$Q/warm.txt" || { echo "PREFLIGHT FAILED (need exit=0 and templates_real=4/4)"; exit 1; }
tr '\n' '\0' < "$Q/tasks.txt" | xargs -0 -P "$NG" -I{} bash -c 'run "$@"' _ {}
echo "END $(date -Is)"
