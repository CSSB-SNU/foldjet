"""One GPU worker of fb_multi.py (started by the parent with CUDA_VISIBLE_DEVICES=k; never run by hand).

The model is loaded once. Targets arrive from the parent over a multiprocessing connection (a receiver thread): every assignment starts its input load
(thread pool) and queues its trunk_seed featurisations for a pool of forked feature processes (fb_data.FeatureDataset logic, at most
`--feature-ahead` results outstanding). The GPU thread runs the targets of its local queue in order with exactly the per-target science and outputs of
the earlier one-process-per-shard runner (grouped trunk_seeds, ranking.csv, run.json, CIFs, done marker). The parent may revoke a queued, not yet started target.

Failures are reported, never retried here: out of memory with CUDA graphs -> the rest of this bucket runs without graphs on this worker and the parent
requeues the target (flagged no-graphs); a sticky CUDA error -> report, flush the writers, exit 75 (the parent respawns the worker).
PYTORCH_CUDA_ALLOC_CONF comes from the parent (the FoldForge default expandable_segments:True unless the user set one).
FF_ALLOW_SIZE_MISMATCH=1 skips the checkpoint blob size check (a local blob of another converter version; off by default)."""

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue as _queue
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--wid", type=int, required=True)
p.add_argument("--addr", required=True)
p.add_argument("--authkey-file", required=True)
p.add_argument("--model", default="af3")
p.add_argument("--checkpoint", default=None, help="a checkpoint file instead of the model's released / converted weights (e.g. a fine-tune's epochNNNN.miniworld.pt)")
p.add_argument("--specs", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seeds", default="42,66,101,2024,8888")
p.add_argument("--nd", type=int, default=5)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--recycles", type=int, default=None)
p.add_argument("--scope", default="trunk", choices=("trunk", "denoiser"))
p.add_argument("--exec-extra", default="{}")
p.add_argument(
    "--release-graphs", default="bucket", choices=("never", "bucket", "target")
)
p.add_argument("--eager", action="store_true")
p.add_argument("--group", type=int, default=5)
p.add_argument(
    "--group-max-bucket",
    type=int,
    default=None,
    help="DEPRECATED (the earlier runner.s rule, residue buckets of 128); ignored unless --group-max-tokens 0",
)
p.add_argument(
    "--group-max-tokens",
    type=int,
    default=512,
    help="trunk_seeds are sampled together only when the target has <= this many AF3 tokens (exact, from the parent's size table); above, one by one",
)
p.add_argument("--max-graphs", type=int, default=32)
p.add_argument("--feature-workers", type=int, default=3)
p.add_argument(
    "--feature-ahead",
    type=int,
    default=6,
    help="featurised (target, trunk_seed) results allowed outstanding (DataLoader: workers x prefetch_factor)",
)
p.add_argument(
    "--feature-timeout",
    type=float,
    default=900.0,
    help="seconds the GPU thread waits for one featurisation before it featurises in-process",
)
p.add_argument("--loader-threads", type=int, default=2)
p.add_argument("--writers", type=int, default=2)
args = p.parse_args()
seeds = [int(x) for x in args.seeds.split(",")]
WID = args.wid
out = Path(args.out)

if os.environ.get("FF_ALLOW_SIZE_MISMATCH"):
    import foldforge.models.checkpoints.lock as _LK

    _LK.check_size = lambda _p: None
import fb_data
import fb_io

import foldforge as ff
from foldforge.models.grouped import GroupedUnsupported
from foldforge.models.io import dense_atoms


def log(msg):
    print(f"MWW {time.strftime('%H:%M:%S')} w{WID} {msg}", flush=True)


# ---------------------------------------------------------------- connection to the parent
conn = Client(args.addr, authkey=Path(args.authkey_file).read_bytes())
_send_lock = threading.Lock()


def send(*msg):
    with _send_lock:
        conn.send((time.time(), *msg))


# ---------------------------------------------------------------- forked helpers (before CUDA and before any thread)
class _ForkPool:
    class _R:
        def __init__(self, r):
            self._r = r

        def done(self):
            return self._r.ready()

        def result(self):
            return self._r.get()

    def __init__(self, n):
        self._p = mp.get_context("fork").Pool(n)

    def submit(self, fn, *a):
        return self._R(self._p.apply_async(fn, a))

    def shutdown(self):
        self._p.close()
        self._p.join()


def _feature_main(jobq, resq):
    """A feature process: (target, trunk_seed, msa_depth, bucketing) jobs -> FeatureDataset items, as a DataLoader worker would make them."""
    torch.set_num_threads(1)
    ds = fb_data.FeatureDataset([], seeds, args.specs, args.model, 0, True)
    while True:
        job = jobq.get()
        if job is None:
            return
        t, s, depth, bucketing, epoch = job
        ds.targets, ds.seeds, ds.msa_depth, ds.bucketing = (
            [t],
            [s],
            int(depth),
            bool(bucketing),
        )
        resq.put((epoch, ds[0]))


dense_atoms.install_msa_feature_cache()  # before the forks: the feature processes inherit it
import alphafold3.common.folding_input  # before the forks: the feature processes start warm
import alphafold3.data.featurisation  # noqa: F401
import torch.multiprocessing as tmp  # tensor-sharing queues, as the DataLoader uses

_fctx = tmp.get_context("fork")
jobq, resq = _fctx.Queue(), _fctx.Queue()
fprocs = []


def _spawn_features():
    for pr in list(fprocs):
        if not pr.is_alive():
            fprocs.remove(pr)
    while len(fprocs) < args.feature_workers:
        pr = _fctx.Process(target=_feature_main, args=(jobq, resq), daemon=True)
        pr.start()
        fprocs.append(pr)


_spawn_features()
writer = _ForkPool(args.writers)
loader = ThreadPoolExecutor(
    max_workers=args.loader_threads, thread_name_prefix="mw-load"
)


def load(t):
    return json.loads(
        (Path(args.specs) / f"{t}.chains.json").read_text()
    ), ff.load_inputs(Path(args.specs) / f"{t}.yaml")


# ---------------------------------------------------------------- local queue, prefetch on assignment
lock = threading.Condition()
local: deque = deque()  # assigned, not started: dicts t, bucket, nographs
loading: dict = {}  # t -> future of load(t)
jobs: deque = deque()  # (t, s) featurisations not yet handed to the feature processes
outstanding = 0  # handed out, result not yet consumed
results: dict = {}  # (t, s) -> item
epoch_of: dict = {}  # t -> assignment number; a result of an older assignment (revoked / failed) is discarded
_epoch = [0]
state = {
    "exit": False,
    "depth": None,
    "bucketing": True,
    "group_cap": {},
    "nographs_buckets": set(),
}


def _pump():
    """Hand queued featurisations to the feature processes up to the outstanding limit (lock held)."""
    global outstanding
    if state["depth"] is None:
        return
    while jobs and outstanding < args.feature_ahead:
        t, s = jobs.popleft()
        jobq.put((t, s, state["depth"], state["bucketing"], epoch_of[t]))
        outstanding += 1


def _on_assign(item):
    t = item["t"]
    _epoch[0] += 1
    epoch_of[t] = _epoch[0]
    local.append(item)
    loading[t] = loader.submit(load, t)
    jobs.extend((t, s) for s in seeds)
    _pump()


def receiver():
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            log("parent connection closed; exiting after the current target")
            with lock:
                state["exit"] = True
                lock.notify_all()
            return
        kind = msg[0]
        with lock:
            if kind == "assign":
                _on_assign(msg[1])
                lock.notify_all()
            elif kind == "init":
                state["group_cap"].update(
                    {int(k): v for k, v in msg[1].get("group_cap", {}).items()}
                )
            elif kind == "revoke":
                t = msg[1]
                it = next((x for x in local if x["t"] == t), None)
                if it is not None:
                    local.remove(it)
                    _forget(t)
                send("revoked", t, it is not None)
            elif kind == "exit":
                state["exit"] = True
                lock.notify_all()


def _forget(t):
    """Drop everything prefetched for t (lock held)."""
    global outstanding
    epoch_of.pop(t, None)
    f = loading.pop(t, None)
    if f is not None:
        f.cancel()
    for j in [j for j in jobs if j[0] == t]:
        jobs.remove(j)
    for k in [k for k in results if k[0] == t]:
        results.pop(k)
        outstanding -= 1
    _pump()


def get_features(t, s):
    """The featurised item of (t, s), or None (featurise in-process) after --feature-timeout or when no feature process is alive."""
    global outstanding
    t0 = time.time()
    while True:
        with lock:
            if (t, s) in results:
                outstanding -= 1
                it = results.pop((t, s))
                _pump()
                return it
        try:
            ep, it = resq.get(timeout=2.0)
        except _queue.Empty:
            it = None
        if it is not None:
            with lock:
                if epoch_of.get(it["target"]) != ep:
                    outstanding -= 1
                    _pump()
                else:
                    results[(it["target"], it["seed"])] = it
            continue
        if time.time() - t0 > args.feature_timeout:
            log(
                f"feature wait for {t} seed {s} > {args.feature_timeout:.0f} s: featurise in-process"
            )
            return None
        if not any(pr.is_alive() for pr in fprocs):
            log("no feature process alive: respawning the pool")
            _spawn_features()


# ---------------------------------------------------------------- model
config = {"precision": "bf16", "diffusion": {"steps": args.steps}}
graphs_on = not args.eager
if not args.eager:
    config["execution"] = {
        "compile": True,
        "cuda_graph": True,
        "bucketing": True,
        "scope": args.scope,
        "max_graphs": args.max_graphs,
        **json.loads(args.exec_extra),
    }
if args.recycles:
    config["trunk"] = {"recycles": args.recycles}

threading.Thread(target=receiver, daemon=True, name="mw-recv").start()
send("hello", WID, os.getpid())
t_all = time.time()
pending: list = []
exit_code = 0
busy_s = 0.0
with ff.load_model(args.model, config=config, checkpoint=args.checkpoint, device="cuda:0") as model:
    from foldforge.api import plan_prediction

    bucketing = bool(model._options.config.execution.bucketing)
    ex = getattr(model, "_execution", None)
    load_s = time.time() - t_all
    log(f"model {args.model} loaded in {load_s:.0f} s")
    # Engine 3.x FP8 (MINIWORLD_FP8 set): log the routes in force; after every trunk_seed group, outside any graph, fp8.check() -- a route
    # that saturated the e4m3 range had its scale raised in place, so the group is run again (at most FP8_REDO times; then kept and logged).
    FP8 = None
    try:
        import miniworld_engine.fp8 as _fp8

        if _fp8.preset():
            FP8 = _fp8
            log(f"FB_FP8 preset={_fp8.preset()} model={_fp8.model()} status={_fp8.status()}")
    except ImportError:
        pass
    FP8_REDO = 3

    gstate = {"res_bucket": None, "graphs": True}

    def res_bucket(inputs):
        residues = sum(
            len(c.sequence)
            for c in inputs.target.chains
            if getattr(c, "sequence", None)
        )
        return (residues + 127) // 128

    def group_limit(bucket, tokens=None):
        """--group when the target has <= --group-max-tokens AF3 tokens, else 1; never above what an out-of-memory taught for this residue bucket or a smaller one."""
        if args.group_max_tokens and tokens is not None:
            cap = args.group if tokens <= args.group_max_tokens else 1
            return min(
                [cap, *(c for b, c in state["group_cap"].items() if b <= bucket)]
            )
        cap = args.group if bucket <= (args.group_max_bucket or 8) else 1
        return min([cap, *(c for b, c in state["group_cap"].items() if b <= bucket)])

    def release_graphs(inputs):
        if args.eager or args.release_graphs == "never" or ex is None:
            return False
        bucket = res_bucket(inputs)
        if args.release_graphs == "bucket" and bucket == gstate["res_bucket"]:
            return False
        gstate["res_bucket"] = bucket
        torch.cuda.synchronize()
        for w in ex.wrappers:
            w.graphs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def set_graphs(on):
        """Turn the CUDA graphs of this process off / on without a restart: every wrapper bypassed (the denoiser still runs compiled), the sampler graph scope detached."""
        if ex is None or args.eager or on == gstate["graphs"]:
            return
        torch.cuda.synchronize()
        if not on:
            for w in ex.wrappers:
                w.graphs.clear()
                w.bypass += 1
            ex.release_sampler_graphs()
            ex.model.foldforge_sampler_scope = None
        else:
            for w in ex.wrappers:
                w.bypass -= 1
            ex.model.foldforge_sampler_scope = ex.sampler_scope
        gstate["graphs"] = on
        gc.collect()
        torch.cuda.empty_cache()

    def finish(item, block=False):
        t, d, futs, meta = item
        if not block and not all(f.done() for f in futs):
            return False
        try:
            rank_rows = [r for f in futs for r in f.result()]
            (d / "ranking.csv").write_text(
                "seed,sample,ranking_score\n"
                + "".join(f"{s},{k},{v:.6f}\n" for s, k, v in sorted(rank_rows))
            )
            timings = meta["timings"]
            (d / "run.json").write_text(
                json.dumps(
                    {
                        "model": args.model,
                        "chains": meta["chains"],
                        "ranking": meta["rank_basis"],
                        "timings_s": timings,
                        "total_s": round(meta["gpu_s"] + meta["load_wait_s"], 1),
                        "gpu_s": round(meta["gpu_s"], 1),
                        "load_wait_s": round(meta["load_wait_s"], 2),
                        "overlap": True,
                        "stages": meta["stages"],
                        "peak_alloc_gib": meta.get("peak_alloc_gib"),
                        "peak_reserved_gib": meta.get("peak_reserved_gib"),
                        "multi": {"worker": WID, "graphs": meta["graphs"]},
                    }
                )
            )
            (d / "done").write_text("ok\n")
            print(
                f"FB_DONE {t}: {meta['chains']} chains, {len(seeds)}x{args.nd} samples, {meta['gpu_s'] + meta['load_wait_s']:.1f} s on the GPU thread "
                f"(load wait {meta['load_wait_s']:.1f} s, trunk {np.mean([x[1] for x in timings]):.1f} s per seed)",
                flush=True,
            )
            send("done", t, round(meta["gpu_s"], 2), round(meta["load_wait_s"], 2))
        except Exception as exc:
            print(f"FB_FAIL {t}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            send("fail", t, "write:" + type(exc).__name__, str(exc)[:300])
        return True

    with lock:
        first = local[0]["t"] if local else None
    probe = ff.load_inputs(Path(args.specs) / f"{first}.yaml") if first else None
    if (
        probe is None
    ):  # nothing assigned yet: any spec of the list gives the cap (it depends on the model, not the target)
        with lock:
            while not local and not state["exit"]:
                lock.wait(1.0)
            first = local[0]["t"] if local else None
        probe = ff.load_inputs(Path(args.specs) / f"{first}.yaml") if first else None
    depth = (
        plan_prediction(
            probe, model=args.model, _options=model._options
        ).request.msa_depth
        if probe is not None
        else 0
    )
    with lock:
        state["depth"], state["bucketing"] = depth, bucketing
        _pump()
    pending_lock = threading.Lock()
    fin_stop = threading.Event()

    def finisher():
        """Closes targets (ranking.csv, run.json, done marker) as soon as their CIF writers finish, not at the next target boundary."""
        while not fin_stop.wait(1.0):
            with pending_lock:
                ready_ = [it for it in pending if all(f.done() for f in it[2])]
                for it in ready_:
                    pending.remove(it)
            for it in ready_:
                finish(it, block=True)

    threading.Thread(target=finisher, daemon=True, name="mw-finish").start()
    send("ready", round(load_s, 1), depth)
    log(
        f"ready: msa_depth {depth}, {args.feature_workers} feature processes x {args.feature_ahead} outstanding"
    )

    while True:
        with lock:
            item = local.popleft() if local else None
            stop = state["exit"]
        if item is None:
            if stop:
                break
            with lock:
                if not local and not state["exit"]:
                    lock.wait(0.5)
            continue
        t = item["t"]
        failure = None
        d = out / t
        nog = bool(item.get("nographs")) or item["bucket"] in state["nographs_buckets"]
        tw = time.time()
        send("start", t, not nog)
        try:
            new_names, inputs = loading.pop(t).result()
            set_graphs(graphs_on and not nog)
            if release_graphs(inputs):
                send("release", t, gstate["res_bucket"])
            torch.cuda.reset_peak_memory_stats()
            load_wait = time.time() - tw
            t0 = time.time()
            d.mkdir(parents=True, exist_ok=True)
            futs, timings, stages = [], [], []
            fetched = set()
            si = 0
            rank_basis = None
            fp8_redo: dict = {}
            while si < len(seeds):
                bucket = res_bucket(inputs)
                g = max(
                    1, min(group_limit(bucket, item.get("tokens")), len(seeds) - si)
                )
                idx = list(range(si, si + g))
                w0, f0 = (
                    dense_atoms.PREFETCH_TIMES["wait"],
                    dense_atoms.PREFETCH_TIMES["featurise_here"],
                )
                try:
                    with ExitStack() as stack:
                        recs = []
                        for k in idx:
                            s = seeds[k]
                            tl = time.time()
                            if k not in fetched:
                                fetched.add(k)
                                fi = get_features(t, s)
                                if fi is not None:
                                    if fi["error"]:
                                        raise RuntimeError(
                                            f"featurisation failed: {fi['error']}"
                                        )
                                    assert (fi["target"], fi["seed"]) == (t, s), (
                                        fi["target"],
                                        fi["seed"],
                                        t,
                                        s,
                                    )
                                    dense_atoms.provide_features_by_signature(
                                        fi["sig"], fi["result"]
                                    )
                            wait_loader = time.time() - tl
                            ts = time.time()
                            trunk = stack.enter_context(model.run_trunk(inputs, seed=s))
                            recs.append((s, trunk, time.time() - ts, wait_loader))
                        t_d0 = time.time()
                        grouped_now = g
                        if g > 1:
                            try:
                                results_ = model.sample_diffusion_group(
                                    [r[1] for r in recs],
                                    samples=args.nd,
                                    steps=args.steps,
                                    seed=recs[0][0],
                                )
                            except GroupedUnsupported as why:
                                log(f"group unsupported ({str(why)[:100]}); one by one")
                                grouped_now = 1
                                results_ = [
                                    model.sample_diffusion(
                                        r[1],
                                        samples=args.nd,
                                        steps=args.steps,
                                        seed=r[0],
                                    )
                                    for r in recs
                                ]
                        else:
                            results_ = [
                                model.sample_diffusion(
                                    recs[0][1],
                                    samples=args.nd,
                                    steps=args.steps,
                                    seed=recs[0][0],
                                )
                            ]
                        t_diff = (time.time() - t_d0) / g
                except torch.OutOfMemoryError:
                    if g == 1:
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    state["group_cap"][bucket] = max(1, g // 2)
                    log(
                        f"FB_GROUP out of memory with {g} trunk_seeds at size bucket {bucket}; now at most {state['group_cap'][bucket]}"
                    )
                    send("groupcap", bucket, state["group_cap"][bucket])
                    continue
                if FP8 is not None and FP8.check():
                    fp8_redo[si] = fp8_redo.get(si, 0) + 1
                    if fp8_redo[si] <= FP8_REDO:
                        log(
                            f"FB_FP8_SATURATED {t} seeds={[r[0] for r in recs]} redo {fp8_redo[si]}/{FP8_REDO} (scales raised)"
                        )
                        del results_, recs
                        continue
                    log(
                        f"FB_FP8_SATURATED_KEPT {t} seeds={[r[0] for r in recs]}: still saturated after {FP8_REDO} redos, kept"
                    )
                for (s, _trunk, t_trunk, wait_loader), res in zip(
                    recs, results_, strict=True
                ):
                    pr = res.prediction
                    if pr.ptm is not None:
                        ptm = pr.ptm.float().cpu().numpy()
                        iptm = (
                            None if pr.iptm is None else pr.iptm.float().cpu().numpy()
                        )
                        rank_basis = "ptm/iptm"
                    else:
                        ptm = pr.plddt.float().mean(-1).cpu().numpy()
                        iptm = None
                        rank_basis = "mean pLDDT (the model returned no pTM)"
                    scores = [
                        float(ptm[k])
                        if iptm is None
                        else float(0.8 * iptm[k] + 0.2 * ptm[k])
                        for k in range(len(res.cifs))
                    ]
                    futs.append(
                        writer.submit(
                            fb_io.write_seed,
                            str(d),
                            t,
                            s,
                            list(res.cifs),
                            scores,
                            new_names,
                        )
                    )
                    timings.append((s, round(t_trunk, 2), round(t_trunk + t_diff, 2)))
                    stages.append(
                        {
                            "seed": s,
                            "trunk": round(t_trunk, 2),
                            "diffusion": round(t_diff, 2),
                            "group": grouped_now,
                            "wait_loader": round(wait_loader, 2),
                            "wait_prefetch": round(
                                (dense_atoms.PREFETCH_TIMES["wait"] - w0) / g, 2
                            ),
                            "featurise_here": round(
                                (dense_atoms.PREFETCH_TIMES["featurise_here"] - f0) / g,
                                2,
                            ),
                        }
                    )
                si += g
                send("progress", t, si)
            gpu_s = time.time() - t0
            busy_s += gpu_s + load_wait
            with pending_lock:
                pending.append(
                    (
                        t,
                        d,
                        futs,
                        {
                            "timings": timings,
                            "chains": len(new_names),
                            "rank_basis": rank_basis,
                            "gpu_s": gpu_s,
                            "load_wait_s": load_wait,
                            "stages": stages,
                            "graphs": gstate["graphs"],
                            "fp8_redo": sum(fp8_redo.values()) if FP8 is not None else None,
                            "peak_alloc_gib": round(
                                torch.cuda.max_memory_allocated() / 2**30, 1
                            ),
                            "peak_reserved_gib": round(
                                torch.cuda.max_memory_reserved() / 2**30, 1
                            ),
                        },
                    )
                )
            send("computed", t, round(gpu_s, 2), round(load_wait, 2))
            if FP8 is not None and hasattr(FP8, "route_stats") and not state.get("fp8_routes_logged"):
                state["fp8_routes_logged"] = True  # once, after the first target: the routes this worker's model takes
                log(f"FB_FP8_ROUTES first_target={t} {json.dumps(FP8.route_stats(), sort_keys=True)}")
        except Exception as exc:
            busy_s += time.time() - tw
            kind = type(exc).__name__
            _c = exc
            while (
                _c is not None and kind != "OutOfMemoryError"
            ):  # a capture wraps the OOM in a RuntimeError: look at the cause chain
                _c = _c.__cause__ or _c.__context__
                if type(_c).__name__ == "OutOfMemoryError":
                    kind = "OutOfMemoryError"
            traceback.print_exc(limit=30)
            failure = (kind, str(exc)[:300].replace("\n", " "))
        if failure is None:
            continue
        # outside the except block: its traceback no longer pins the failed target's GPU tensors
        kind, msg = failure
        with lock:
            _forget(t)
        if kind == "OutOfMemoryError":
            with_graphs = gstate["graphs"]
            if with_graphs:
                state["nographs_buckets"].add(item["bucket"])
                set_graphs(False)
            gc.collect()
            torch.cuda.empty_cache()
            log(
                f"FB_OOM {t} graphs={with_graphs}: the rest of bucket {item['bucket']} runs without CUDA graphs here"
            )
            send("fail", t, "oom_graphs" if with_graphs else "oom", msg)
            continue
        send("fail", t, kind, msg)
        if kind == "AcceleratorError":
            log("sticky CUDA error: flushing writers, exit 75")
            exit_code = 75
            break
    fin_stop.set()
    with pending_lock:
        rest = list(pending)
        pending.clear()
    for it in rest:
        finish(it, block=True)
    if FP8 is not None and hasattr(FP8, "route_stats"):  # engine >= 3.0.2: which FP8 routes fired (eager / capture-time counts; replays do not count)
        log(f"FB_FP8_ROUTES {json.dumps(FP8.route_stats(), sort_keys=True)}")
    send("bye", round(busy_s, 1), exit_code)
for _ in fprocs:
    jobq.put(None)
writer.shutdown()
loader.shutdown(wait=False, cancel_futures=True)
log(
    f"FB_PREFETCH {dense_atoms.PREFETCH_STATS}; end {time.time() - t_all:.0f} s exit {exit_code}"
)
sys.stdout.flush()
os._exit(exit_code)
