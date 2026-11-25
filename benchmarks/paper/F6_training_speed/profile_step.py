#!/usr/bin/env python
"""F6 step profiler: where does ONE warm optimizer step of the F6 training benchmark spend its time?

Runs FoldJet's own ``scripts/train_protenix1.py`` (the F6 harness: ``--bench`` stand-in pool, ``--step-graph``, activation checkpointing, 48 augmentations in one pass,
``af3_default`` precision) unchanged and wraps ``foldjet.train.step.micro_step``:

* every micro step after the warm-up step is timed on its own (synchronised before and after; wall seconds, recycle count);
* the micro steps of the profiled step (``--prof-step``, default 1: step 0 holds the graph captures / warm-up) run under ``torch.profiler`` (CPU + CUDA); each micro step is
  wrapped in a ``record_function`` range so that the kernels can be assigned to it;
* the Chrome trace is parsed here (no extra tools): GPU busy time (union of kernel intervals) against wall time, kernel count, the number of launches the host issued
  (``cudaGraphLaunch`` = graph replay, ``cudaLaunchKernel`` = eager), the top kernels, kernel classes (the engine's ``ieee_f32`` FFMA chain, cuBLAS, Triton, PyTorch
  elementwise / reduction ...), the share of kernels shorter than 5 / 10 / 20 us.

Usage (the launcher ``profile_step.sh`` builds the whole command line):
    python profile_step.py [--prof-step N] [--out DIR] [--keep-trace] -- <train_protenix1.py arguments>
    python profile_step.py --analyze TRACE.json [--out DIR]       # trace of an earlier run (CPU only, no GPU needed)
    python profile_step.py --probe                                 # GEMM rate probe: fp32 IEEE (FFMA) / TF32 / bf16 on this card, the engine's ieee_f32 GEMM

Never run on a login node (srun / sbatch; on the B200 host through the lane driver).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import runpy
import statistics
import subprocess
import sys
import time
from pathlib import Path

# kernel classes, first match wins (lower-cased kernel name); the engine's own kernels that match nothing are 'engine_or_other'
CLASSES: list[tuple[str, str]] = [
    ("ieee_f32 (engine FFMA chain)", r"gemm_k<|gemm_k\b|sum_slabs_k|reduce_slabs_k|ln_fwd_k|ln_bwd_k"),
    ("triton", r"^triton_|^void triton|triton_(poi|red|per|tem|spl)"),
    ("cublas fp32 simt (ffma)", r"sgemm|simt|ampere_s|maxwell_s|volta_s"),
    ("cublas / cutlass tensor-core gemm", r"xmma|cutlass|nvjet|cublaslt|wgmma|tensorop|s16816|s1688|h16816|bf16.*gemm|gemm.*bf16|gemvx|gemv"),
    ("memcpy / memset", r"^memcpy|^memset|\[memcpy|\[memset"),
    ("pytorch elementwise / copy", r"elementwise_kernel|vectorized|unrolled_elementwise|copy_kernel|catarray|index_|fill_|where|clamp|cat_|gather|scatter"),
    ("pytorch reduce / norm / softmax", r"reduce_kernel|layer_norm|softmax|batch_norm|var_mean|cunn_|norm_"),
]

KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def classify(name: str, cat: str) -> str:
    low = name.lower()
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "memcpy / memset"
    for label, rx in CLASSES:
        if re.search(rx, low):
            return label
    return "engine_or_other (cuda: trimul/triattn/dit/pwa/opm/ln ...)"


def short(name: str, width: int = 120) -> str:
    name = re.sub(r"\s+", " ", name)
    return name if len(name) <= width else name[: width - 3] + "..."


def union_us(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of [start, end) intervals (us)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def analyze_trace(path: str, out: Path, tag: str = "") -> dict:
    """Parse a Chrome trace of the profiled micro steps; print and save the tables. Returns the summary dict."""
    with open(path) as fh:
        events = json.load(fh).get("traceEvents", [])
    wins = sorted(
        ((e["ts"], e["ts"] + e["dur"], e["name"]) for e in events if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("F6MICRO")),
        key=lambda w: w[0],
    )
    kern = [e for e in events if e.get("cat") in KERNEL_CATS and "dur" in e and "ts" in e]
    runtime = [e for e in events if e.get("cat") == "cuda_runtime" and "ts" in e]
    if not wins:  # no ranges: the whole trace is one window
        lo = min((e["ts"] for e in kern), default=0.0)
        hi = max((e["ts"] + e["dur"] for e in kern), default=0.0)
        wins = [(lo, hi, "F6MICRO_all")]
    # kernels of a window = those starting inside it
    per_win = []
    by_name: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    by_class: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
    durs: list[float] = []
    all_iv: list[tuple[float, float]] = []
    for lo, hi, name in wins:
        ks = [e for e in kern if lo <= e["ts"] <= hi]
        iv = [(e["ts"], e["ts"] + e["dur"]) for e in ks]
        busy = union_us(iv)
        span = (max(b for _, b in iv) - min(a for a, _ in iv)) if iv else 0.0
        rt = collections.Counter(e["name"] for e in runtime if lo <= e["ts"] <= hi and e["name"].startswith(("cudaGraphLaunch", "cudaLaunchKernel", "cuLaunchKernel", "cudaMemcpyAsync", "cudaMemsetAsync", "cuLaunchKernelEx", "cudaLaunchKernelExC")))
        per_win.append(
            {
                "range": name,
                "wall_ms": (hi - lo) / 1e3,
                "gpu_busy_ms": busy / 1e3,
                "gpu_span_ms": span / 1e3,
                "n_kernels": len(ks),
                "sum_kernel_ms": sum(e["dur"] for e in ks) / 1e3,
                "host_launches": dict(rt),
            }
        )
        all_iv += iv
        for e in ks:
            k = by_name[short(e["name"])]
            k[0] += 1
            k[1] += e["dur"]
            c = by_class[classify(e["name"], e["cat"])]
            c[0] += 1
            c[1] += e["dur"]
            durs.append(e["dur"])
    wall = sum(w["wall_ms"] for w in per_win)
    busy = sum(w["gpu_busy_ms"] for w in per_win)
    span = sum(w["gpu_span_ms"] for w in per_win)
    n = sum(w["n_kernels"] for w in per_win)
    tot_us = sum(v[1] for v in by_name.values())
    graph_launch = sum(v for w in per_win for k, v in w["host_launches"].items() if k.startswith("cudaGraphLaunch"))
    eager_launch = sum(v for w in per_win for k, v in w["host_launches"].items() if k.startswith(("cudaLaunchKernel", "cuLaunchKernel")))
    lines = []
    p = lines.append
    p(f"=== F6 STEP PROFILE {tag} ({len(per_win)} micro steps of one optimizer step)")
    p(f"{'range':<22}{'wall ms':>10}{'gpu busy ms':>13}{'busy/wall':>10}{'gpu span ms':>13}{'busy/span':>10}{'kernels':>9}{'sum dur ms':>12}  host launches")
    for w in per_win:
        p(f"{w['range']:<22}{w['wall_ms']:>10.1f}{w['gpu_busy_ms']:>13.1f}{w['gpu_busy_ms'] / max(w['wall_ms'], 1e-9):>10.3f}{w['gpu_span_ms']:>13.1f}{w['gpu_busy_ms'] / max(w['gpu_span_ms'], 1e-9):>10.3f}{w['n_kernels']:>9d}{w['sum_kernel_ms']:>12.1f}  {w['host_launches']}")
    p(f"{'TOTAL':<22}{wall:>10.1f}{busy:>13.1f}{busy / max(wall, 1e-9):>10.3f}{span:>13.1f}{busy / max(span, 1e-9):>10.3f}{n:>9d}{tot_us / 1e3:>12.1f}")
    p(f"GPU idle fraction of the micro-step wall (1 - busy/wall): {1 - busy / max(wall, 1e-9):.3f};  of the GPU span (gaps between kernels): {1 - busy / max(span, 1e-9):.3f}")
    p(f"host launches: cudaGraphLaunch {graph_launch}, cudaLaunchKernel (eager) {eager_launch}  ->  {'GRAPH REPLAY' if graph_launch and eager_launch < 50 * max(graph_launch, 1) else 'EAGER (or mixed)'}")
    if durs:
        sd = sorted(durs)
        p(f"kernel duration: mean {statistics.fmean(durs):.2f} us, median {sd[len(sd) // 2]:.2f} us, p10 {sd[len(sd) // 10]:.2f} us; share of kernel COUNT < 5 us {sum(d < 5 for d in durs) / len(durs):.3f}, < 10 us {sum(d < 10 for d in durs) / len(durs):.3f}, < 20 us {sum(d < 20 for d in durs) / len(durs):.3f}; share of kernel TIME in kernels < 10 us {sum(d for d in durs if d < 10) / sum(durs):.3f}, < 20 us {sum(d for d in durs if d < 20) / sum(durs):.3f}")
    p("")
    p(f"{'kernel class':<62}{'kernels':>9}{'ms':>10}{'% of kernel time':>18}{'% of wall':>11}")
    for cl, (c, us) in sorted(by_class.items(), key=lambda kv: -kv[1][1]):
        p(f"{cl:<62}{int(c):>9d}{us / 1e3:>10.1f}{100 * us / max(tot_us, 1e-9):>18.2f}{100 * us / 1e3 / max(wall, 1e-9):>11.2f}")
    p("")
    p("top 25 kernels by total time")
    p(f"{'ms':>9}{'%':>7}{'count':>8}{'avg us':>9}  kernel")
    for name, (c, us) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:25]:
        p(f"{us / 1e3:>9.2f}{100 * us / max(tot_us, 1e-9):>7.2f}{int(c):>8d}{us / c:>9.1f}  {name}")
    text = "\n".join(lines)
    print(text, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"profile_tables{tag}.txt").write_text(text + "\n")
    summary = {
        "micro_steps": per_win,
        "wall_ms": wall,
        "gpu_busy_ms": busy,
        "gpu_idle_fraction": 1 - busy / max(wall, 1e-9),
        "n_kernels": n,
        "graph_launches": graph_launch,
        "eager_launches": eager_launch,
        "classes": {k: {"kernels": int(v[0]), "ms": v[1] / 1e3} for k, v in by_class.items()},
        "top_kernels": [{"name": k, "count": int(v[0]), "ms": v[1] / 1e3} for k, v in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:60]],
    }
    (out / f"profile_summary{tag}.json").write_text(json.dumps(summary, indent=1))
    return summary


def gemm_probe() -> None:
    """GEMM rates of this card: cuBLAS fp32 with TF32 denied (FFMA), TF32, bf16; the engine's ieee_f32 GEMM at the pair-transition shapes."""
    import torch

    dev = torch.device("cuda")
    print(f"PROBE gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} cc={torch.cuda.get_device_capability(0)}", flush=True)

    def timeit(fn, flop):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            fn()
        b.record()
        torch.cuda.synchronize()
        ms = a.elapsed_time(b) / 10
        return ms, flop / ms / 1e9  # TFLOP/s

    shapes = {"square 8192^3": (8192, 8192, 8192), "pair transition up (M=147456 K=128 N=512)": (147456, 128, 512), "pair transition down (M=147456 K=512 N=128)": (147456, 512, 128), "wgrad (M=128 K=147456 N=512)": (128, 147456, 512)}
    for label, (m, k, n) in shapes.items():
        a32 = torch.randn(m, k, device=dev)
        b32 = torch.randn(k, n, device=dev)
        flop = 2.0 * m * n * k
        res = {}
        for mode, prec, dt in (("fp32 IEEE (cuBLAS simt FFMA)", "highest", torch.float32), ("TF32", "high", torch.float32), ("bf16", "highest", torch.bfloat16)):
            torch.set_float32_matmul_precision(prec)
            torch.backends.cuda.matmul.allow_tf32 = prec != "highest"
            x, y = a32.to(dt), b32.to(dt)
            res[mode] = timeit(lambda x=x, y=y: x @ y, flop)
        try:
            from foldjet_engine.kernels import ieee_f32

            if ieee_f32.available(a32, hopper=True) and m % 4 == 0 and n % 4 == 0 and k % 4 == 0:
                res["engine ieee_f32.mm (FFMA)"] = timeit(lambda: ieee_f32.mm(a32, b32), flop)
        except Exception as exc:  # noqa: BLE001 - a probe
            res["engine ieee_f32.mm (FFMA)"] = (float("nan"), float("nan"))
            print(f"PROBE ieee_f32 unavailable: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        for mode, (ms, tf) in res.items():
            print(f"PROBE {label:<48} {mode:<30} {ms:9.3f} ms {tf:9.1f} TFLOP/s", flush=True)
        del a32, b32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    # a pure FFMA loop: the SM FP32 rate without memory
    print("PROBE nvidia-smi", subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip(), flush=True)


def _install_random_init() -> None:
    """PROFILE-ONLY (F6PROF_RANDOM_INIT=1, rented third-party machine: no weights uploaded): replace the checkpoint load by a random initialisation.

    Kernel times do not depend on the values (except data-dependent guard paths). Pass any existing file named *.miniworld.pt as --ckpt (it is never read).
    """
    import torch
    from foldjet.models.checkpoints import miniworld

    def load_into(model, path, *, family):  # noqa: ARG001
        g = torch.Generator().manual_seed(0)
        fourier = model.get_submodule("diffusion_head.fourier_embeddings")
        for name, init in (("weight", torch.randn(256, generator=g)), ("bias", torch.rand(256, generator=g))):
            fourier.register_buffer(name, init)
        with torch.no_grad():
            for pname, p in model.named_parameters():
                if p.ndim >= 2:
                    p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype) / (p.shape[-1] ** 0.5))
                elif pname.endswith(("weight", "scale", "gamma")):
                    p.fill_(1.0)
                else:
                    p.zero_()
        print("F6PROF RANDOM_INIT: parameters randomly initialised (no checkpoint read)", flush=True)
        return {"format": "random_init", "format_version": 0, "source": "random", "mapped_records": 0}

    miniworld.load_into = load_into


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--prof-step", type=int, default=1, help="optimizer step whose micro steps are profiled (0 = capture/warm-up step)")
    ap.add_argument("--out", default=".", help="output directory (tables, summary, trace)")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    ap.add_argument("--keep-trace", action="store_true")
    ap.add_argument("--analyze", default=None, help="analyse an existing Chrome trace and exit (no GPU)")
    ap.add_argument("--probe", action="store_true", help="GEMM rate probe and exit")
    ap.add_argument("--train-script", default=None)
    ap.add_argument("train_args", nargs=argparse.REMAINDER, help="arguments of scripts/train_protenix1.py after --")
    args = ap.parse_args()
    out = Path(args.out)
    if args.analyze:
        analyze_trace(args.analyze, out, args.tag)
        return
    if args.probe:
        gemm_probe()
        return
    targs = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    accum = int(targs[targs.index("--grad-accum") + 1]) if "--grad-accum" in targs else 8
    lo, hi = accum * args.prof_step, accum * (args.prof_step + 1)

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from foldjet.train import step as step_mod

    orig = step_mod.micro_step
    state = {"n": 0, "prof": None, "walls": []}

    def wrapped(model, example, cfg, passes=None, scale=1.0, timer=None, **kw):
        idx = state["n"]
        state["n"] += 1
        if idx < accum:  # the warm-up step (graph captures / first calls): untouched
            return orig(model, example, cfg, passes=passes, scale=scale, timer=timer, **kw)
        profiling = lo <= idx < hi
        if idx == lo:
            state["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False)
            torch.cuda.synchronize()
            state["prof"].start()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if profiling:
            with record_function(f"F6MICRO{idx}_passes{passes}"):
                res = orig(model, example, cfg, passes=passes, scale=scale, timer=timer, **kw)
                torch.cuda.synchronize()
        else:
            res = orig(model, example, cfg, passes=passes, scale=scale, timer=timer, **kw)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        state["walls"].append((idx, passes, wall, profiling))
        print(f"F6PROF micro={idx} step={idx // accum} passes={passes} wall_s={wall:.4f} profiled={int(profiling)}", flush=True)
        if idx == hi - 1:
            prof = state["prof"]
            prof.stop()
            out.mkdir(parents=True, exist_ok=True)
            trace = out / f"trace{args.tag}.json"
            prof.export_chrome_trace(str(trace))
            try:
                print("F6PROF CPU ops by self CPU time:\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=70), flush=True)
                print("F6PROF ops by self device time (eager runs attribute kernels to ops; a graph replay does not):\n" + prof.key_averages().table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=70), flush=True)
                ops = [
                    {"op": e.key, "count": e.count, "self_device_ms": getattr(e, "self_device_time_total", 0.0) / 1e3, "self_cpu_ms": e.self_cpu_time_total / 1e3}
                    for e in prof.key_averages()
                    if not e.key.startswith("F6MICRO")
                ]
                ops.sort(key=lambda o: -o["self_device_ms"])
                (out / f"ops{args.tag}.json").write_text(json.dumps(ops[:150], indent=1))
            except Exception as exc:  # noqa: BLE001
                print(f"F6PROF key_averages failed: {exc}", flush=True)
            analyze_trace(str(trace), out, args.tag)
            if not args.keep_trace:
                trace.unlink(missing_ok=True)
            by_passes = collections.defaultdict(list)
            for _, p, w, _ in state["walls"]:
                by_passes[p].append(w)
            print("F6PROF micro wall by recycle count (all timed micro steps): " + "; ".join(f"passes={p}: n={len(v)} median {statistics.median(v):.4f} s" for p, v in sorted(by_passes.items())), flush=True)
        return res

    if os.environ.get("F6PROF_RANDOM_INIT"):
        _install_random_init()
    step_mod.micro_step = wrapped
    script =args.train_script or str(Path(__file__).resolve().parents[3] / "scripts" / "train_protenix1.py")
    sys.argv = [script, *targs]
    runpy.run_path(script, run_name="__main__")
    if os.environ.get("F6PROF_PROBE_AFTER"):
        gemm_probe()


if __name__ == "__main__":
    main()
