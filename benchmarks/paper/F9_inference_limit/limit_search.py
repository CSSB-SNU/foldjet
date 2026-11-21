"""F9 driver: the largest 5I28 assembly (n copies x 128 tokens) one forward completes on this GPU, for ONE predictor and ONE arm. Resumable.

    python limit_search.py --model af3 --arm engine|engine_fp8|torch --out DIR [--steps 16 20 24 32 ...] [--probe-after 2]
    python limit_search.py --kit af3_torch --out DIR            (the Anthropic kit through the 5I28 kit harness: fast, big only on a fast OOM)

Search: the sizes of --steps in order, one probe = one fresh process (`run_size.py`); at the first size that does not complete (out of memory) bisect by one
copy between the last size that completed and that size; then probe --probe-after more step sizes past the edge to catch a non-monotonic edge (the RTX 5090
showed one with AF3: 19-22 out of memory, 23-24 ok before the pair-band fix). A failure that is not memory (non-finite output, a bucket / ladder stop, a crash)
ends the search and is reported with its reason. An out-of-memory error of the 'reserved but unallocated' kind is retried ONCE with expandable segments
(the 5I28 harness's own last fallback). Every probe is a line of DIR/sizes.jsonl (probes already there are not repeated); DIR/limit.json = the summary.
Peak memory is recorded per probe (model forward peak and process peak).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import ffpaper  # noqa: E402

DEFAULT_STEPS = [16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 128]


def ff_probe(a, n: int) -> dict:
    cell_dir = a.out / "cells"
    cmd = [sys.executable, str(HERE / "run_size.py"), "--model", a.model, "--arm", a.arm, "--n", str(n), "--out", str(cell_dir)]
    if a.exec_mode:
        cmd += ["--exec", a.exec_mode]
    cell_path = cell_dir / f"n{n}.json"
    t0 = time.time()
    with (a.out / "run.log").open("a") as log:
        for attempt in range(2):
            cell_path.unlink(missing_ok=True)
            env = dict(os.environ)
            if attempt:
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log.write(f"=== probe n={n} attempt={attempt} {time.strftime('%F %T')}\n")
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)  # noqa: S603
            cell = json.loads(cell_path.read_text()) if cell_path.exists() else {"status": "failed", "error": "no cell written (process died)"}
            if not (cell.get("status") == "oom" and cell.get("fragmented") and attempt == 0 and "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")):
                break
            cell["first_attempt"] = "oom (fragmented); retried once with expandable segments"
    cell["retried_expandable"] = attempt == 1
    cell["probe_wall_s"] = round(time.time() - t0, 1)
    return cell


def kit_probe(a, n: int) -> dict:
    """One kit cell through the harness's fb.sh (fast; big on a fast OOM). Needs the kit input of this n (kit_inputs.py builds it)."""
    kit_env = dict(os.environ)
    bench = Path(kit_env["KIT_BENCH"])
    sys.path.insert(0, str(HERE))
    import kit_inputs  # noqa: E402

    t0 = time.time()
    built = kit_inputs.ensure(a.kit, n, bench)
    if built is not None:
        return {"status": "no_input", "error": built, "n": n, "tokens": 128 * n}
    cmd, log = kit_inputs.command(a.kit, n, bench)
    p = subprocess.run(["bash", str(bench / "fb.sh"), "1", log, "bash", "-c", cmd], capture_output=True, text=True, check=False, env=kit_env)  # noqa: S603, S607
    last = (p.stdout.strip().splitlines() or [""])[-1]
    fwd = re.search(r"fwd=\[([^\]]*)\]", last)
    forwards = [float(x) for x in re.split(r"[ ,]+", fwd.group(1).strip()) if x] if fwd else []
    text = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    if forwards:
        status = "ok"
    elif "exit=2" in last:
        status = "no_input"
    elif re.search(r"RESOURCE_EXHAUSTED|[Oo]ut of memory|OutOfMemory", text):
        status = "oom"
    else:
        status = "failed"
    return {"n": n, "tokens": 128 * n, "status": status, "mode": "big" if "fast=oom" in last else "fast", "kit_line": last[:400], "forwards_s": forwards,
            "probe_wall_s": round(time.time() - t0, 1), "error": "" if status == "ok" else last[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=ffpaper.PREDICTORS_5I28)
    ap.add_argument("--arm", choices=["engine", "engine_fp8", "torch"])
    ap.add_argument("--exec", dest="exec_mode", choices=["benchmark", "eager"])
    ap.add_argument("--kit")
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    ap.add_argument("--probe-after", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if bool(a.model) == bool(a.kit):
        ap.error("exactly one of --model (FoldForge) or --kit (Anthropic kit)")
    if a.model and not a.arm:
        ap.error("--arm is required with --model")
    a.out.mkdir(parents=True, exist_ok=True)
    jl = a.out / "sizes.jsonl"
    results: dict[int, dict] = {r["n"]: r for r in ffpaper.read_jsonl(jl)}

    def run(n: int) -> dict:
        if n not in results:
            r = kit_probe(a, n) if a.kit else ff_probe(a, n)
            r["n"], r["tokens"] = n, 128 * n
            results[n] = r
            ffpaper.append_jsonl(jl, r)
            print(f"PROBE n={n} tokens={128 * n} {r['status']} peak_alloc={r.get('model_peak_alloc_gib')} process_peak={r.get('process_peak_alloc_gib')} "
                  f"{(r.get('error') or '')[:140]}", flush=True)
        return results[n]

    last_ok, first_bad, stop_reason = 0, None, ""
    for n in a.steps:
        r = run(n)
        if r["status"] == "ok":
            last_ok = n
            continue
        first_bad = n
        if r["status"] != "oom":
            stop_reason = f"{r['status']}: {r.get('error', '')}"[:300]
        break
    if first_bad is not None and not stop_reason:
        lo, hi = last_ok, first_bad
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if run(mid)["status"] == "ok":
                lo = mid
            else:
                hi = mid
        last_ok, first_bad = lo, hi
        for n in [s for s in a.steps if s > first_bad][: a.probe_after]:
            run(n)
    ok_sizes = sorted(n for n, r in results.items() if r["status"] == "ok")
    edge = results.get(last_ok, {})
    summary = {
        "predictor": a.model or a.kit, "arm": a.arm or "kit",
        "largest_ok_contiguous": last_ok or None, "tokens_ok_contiguous": 128 * last_ok if last_ok else None,
        "largest_ok_any": ok_sizes[-1] if ok_sizes else None, "first_not_ok": first_bad,
        "first_not_ok_status": results[first_bad]["status"] if first_bad in results else None,
        "non_monotonic": bool(ok_sizes and first_bad and ok_sizes[-1] > first_bad),
        "peak_alloc_gib_at_edge": edge.get("model_peak_alloc_gib"), "process_peak_alloc_gib_at_edge": edge.get("process_peak_alloc_gib"),
        "stop_reason": stop_reason, "tried": sorted(results),
    }
    (a.out / "limit.json").write_text(json.dumps(summary, indent=1))
    print("LIMIT " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
