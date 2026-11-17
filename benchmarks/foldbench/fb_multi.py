"""FoldBench end to end on N GPUs with one command: a dynamic, bucket-affine scheduler over one worker process per GPU.

  run_py.sh fb_multi.py MODEL TARGETS_FILE SPEC_DIR OUT_DIR [--gpus 0,1] [--lookahead 2] [--order total|largest] [worker args, e.g. --recycles 11]

Parent (this process, no CUDA): sorts the list by exact size bucket (fast_sort: BucketShape.select(tokens, atoms, 1)), estimates a cost per target
(fb_sched.cost_model), and serves work to the workers (fb_multi_worker.py, CUDA_VISIBLE_DEVICES=k) over a unix-socket connection:
  - assignment (fb_sched.Scheduler): bucket affinity -> claim the unclaimed bucket with the largest remaining cost -> share (steal) the bucket with the
    largest remaining cost; every worker holds its running target + `--lookahead` assigned targets (inputs load + trunk_seed featurisation start on
    assignment); a worker that runs dry while another still has a queued (not started) target gets it revoked from that worker.
  - failures: a target failing `--max-fails` times (default 2) gets `<t>/failed/marker` + a failures.tsv line; out of memory with CUDA graphs ->
    the target is retried without graphs (that worker runs the rest of that bucket without graphs); a dead worker (exit, signal, `--stall` seconds
    without a message while running a target) is killed with its process group, its in-flight targets requeued, and it is respawned.
  - resumable: targets with `<t>/done` (or `<t>/failed/marker`) are skipped.
  - small concurrency (`--small-concurrency K`, default 1 = off): the parent starts one CUDA MPS daemon for the job (node-local pipe dir, stopped at
    the end; `--mps off` = plain time slicing, measured 2x less gain) and each GPU gets up to K workers while it runs small targets (token bucket <=
    `--small-max-tokens`, default 512). Slots 1..K-1 of a GPU (wid = G*slot + gpu index) take small targets only and start on demand: once the
    GPU's primary runs / holds only small targets, no unclaimed large bucket is left and >= `--small-min-left-s` estimated seconds of small work per
    GPU are unassigned. While a slot is alive the primary takes small targets only; when no small target is left to assign but a large one is, the
    slots run what they hold and exit, then the primary runs the large target alone. Large buckets are claimed before
    small ones (K > 1). Outputs are byte-identical to K = 1 (measured on 40 targets).
Per target: CIFs, ranking.csv, run.json, done (README.md). Parent events: OUT/multi_events.log; per-target rows:
OUT/multi_targets.tsv; summary: OUT/multi_summary.json; worker logs: OUT/workers/w{k}_{incarnation}.log.
PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True (the FoldForge default, applied at `import foldforge`; a value set by the user is kept)."""

import argparse
import json
import os
import queue as _queue
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from multiprocessing.connection import Listener, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fb_sched

import foldforge  # noqa: F401 - applies the FoldForge allocator default (expandable_segments), inherited by the workers

ALLOC = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", "")
)

p = argparse.ArgumentParser(allow_abbrev=False)
p.add_argument("model")
p.add_argument("targets")
p.add_argument("specs")
p.add_argument("out")
p.add_argument(
    "--gpus",
    default=None,
    help="comma list of devices (default: CUDA_VISIBLE_DEVICES, else 0)",
)
p.add_argument(
    "--lookahead",
    type=int,
    default=2,
    help="targets assigned ahead of the running one per worker",
)
p.add_argument(
    "--order",
    default="total",
    choices=("total", "largest"),
    help="claim the bucket with the largest remaining summed cost, or with the largest single target",
)
p.add_argument("--max-fails", type=int, default=2)
p.add_argument("--max-respawns", type=int, default=8, help="per GPU")
p.add_argument(
    "--stall",
    type=float,
    default=1800.0,
    help="seconds without any message from a worker running a target -> killed + respawned",
)
p.add_argument("--startup-timeout", type=float, default=2400.0)
p.add_argument(
    "--per-target-tsv",
    default=str(HERE / "foldbench_cost_s.tsv"),
    help="measured per-target seconds for the cost estimate (target, af3_tokens, seconds)",
)
p.add_argument(
    "--calib-runs",
    default="",
    help="comma list of earlier output dirs whose run.json total_s calibrate the cost estimate (read-only; empty = none)",
)
p.add_argument(
    "--simulate",
    action="store_true",
    help="dry run of the schedule with the estimated costs (CPU only), then exit",
)
p.add_argument(
    "--kill-test",
    default=None,
    help="SECONDS:WID: SIGKILL that worker's python once, SECONDS after the start (requeue/respawn test)",
)
p.add_argument(
    "--small-concurrency",
    type=int,
    default=1,
    help="workers per GPU for small targets (token bucket <= --small-max-tokens); 1 = one worker per GPU (unchanged)",
)
p.add_argument(
    "--small-max-tokens",
    type=int,
    default=512,
    help="a target is small when its token bucket is <= this (512 = exactly the targets with <= 512 AF3 tokens)",
)
p.add_argument(
    "--small-min-left-s",
    type=float,
    default=120.0,
    help="extra small workers of a GPU are (re)started only while the unassigned small targets hold at least this many estimated seconds per GPU (a slot costs ~30 s to load)",
)
p.add_argument(
    "--mps",
    default="auto",
    choices=("auto", "on", "off"),
    help="CUDA MPS daemon for the job (auto: on when --small-concurrency > 1); a failed start falls back to no MPS",
)
args, worker_args = p.parse_known_args()
args.specs = str(
    Path(args.specs).resolve()
)  # the workers run in their own working directory
if "--recycles" not in worker_args:
    print(
        "MW note: no --recycles given; the FoldBench runs use --recycles 11", flush=True
    )

out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
(out / "workers").mkdir(exist_ok=True)
T0 = time.time()
_evf = open(out / "multi_events.log", "a", buffering=1)


def ev(event, **kw):
    line = (
        f"MW {time.strftime('%Y-%m-%dT%H:%M:%S')} +{time.time() - T0:9.1f} {event:9s} "
        + " ".join(f"{k}={v}" for k, v in kw.items())
    )
    print(line, flush=True)
    _evf.write(line + "\n")


# ---------------------------------------------------------------- the list, sorted + bucketed, with costs
all_targets = Path(args.targets).read_text().split()
todo = [
    t
    for t in all_targets
    if not (out / t / "done").exists() and not (out / t / "failed" / "marker").exists()
]
import fast_sort

fast_sort.size_sort(todo, args.specs, 32, out / "sort_order.tsv") if todo else None
_sizes = (
    {
        l.split("\t")[0]: int(l.split("\t")[1])
        for l in fast_sort.SIZES.read_text().splitlines()[1:]
    }
    if fast_sort.SIZES.exists()
    else {}
)
est, cost_desc = fb_sched.cost_model(
    Path(args.per_target_tsv), [d for d in args.calib_runs.split(",") if d], _sizes
)
items = []
for line in (out / "sort_order.tsv").read_text().splitlines() if todo else []:
    t, tok, at, tb, ab = line.split("\t")
    items.append(
        {
            "t": t,
            "tokens": int(tok),
            "atoms": int(at),
            "bucket": (int(tb), int(ab)),
            "cost": round(est(t, int(tok)), 2),
        }
    )
n_buckets = len({it["bucket"] for it in items})
ev(
    "list",
    targets=len(all_targets),
    todo=len(todo),
    skipped=len(all_targets) - len(todo),
    buckets=n_buckets,
    est_total_s=round(sum(i["cost"] for i in items)),
    cost=f"'{cost_desc}'",
    alloc=ALLOC or "-",
)
gpus = (args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")

if args.simulate:
    for order in ("total", "largest"):
        r = fb_sched.simulate(items, len(gpus), order)
        print(
            f"SIM order={order}: makespan {r['makespan_s'] / 3600:.2f} h (ideal {r['ideal_s'] / 3600:.2f} h), finish spread {r['spread_s'] / 60:.1f} min, "
            f"per-GPU finish {[round(x / 3600, 2) for x in r['finish_s']]} h, picks {r['picks']}",
            flush=True,
        )
    sys.exit(0)
if not items:
    ev("end", msg="nothing to do")
    sys.exit(0)

sched = fb_sched.Scheduler(
    items,
    args.order,
    first=(lambda b: b[0] > args.small_max_tokens)
    if args.small_concurrency > 1
    else None,
)  # K > 1: large buckets first
info = {it["t"]: it for it in items}

# ---------------------------------------------------------------- workers
authkey = secrets.token_bytes(32)
akf = out / "workers" / ".authkey"
akf.write_bytes(authkey)
os.chmod(akf, 0o600)
addr = f"/tmp/fbmulti_{os.getpid()}_{secrets.token_hex(4)}.sock"
listener = Listener(addr, family="AF_UNIX", authkey=authkey)
incoming: _queue.Queue = _queue.Queue()


def _accept():
    while True:
        try:
            incoming.put(listener.accept())
        except Exception:  # closed at the end, or a bad handshake
            if getattr(listener, "_closed", False):
                return
            time.sleep(0.1)


threading.Thread(target=_accept, daemon=True).start()


class W:
    def __init__(self, wid, gpu, slot=0):
        self.wid, self.gpu, self.slot = (
            wid,
            gpu,
            slot,
        )  # slot 0: the GPU's primary worker (any target); slot > 0: small targets only
        self.retiring = False
        self.crashed = False
        self.inc = 0
        self.respawns = 0
        self.popen = None
        self.conn = None
        self.alive = False
        self.ready = False
        self.queue: list = []  # assigned, not started
        self.current = None  # (t, start_ts)
        self.computed: set = set()  # computed, done marker not yet written
        self.last_msg = 0.0
        self.spawn_t = 0.0
        self.load_s = 0.0
        self.busy_s = 0.0
        self.last_finish = None
        self.first_ready = None
        self.revoke_pending = False
        self.n_done = 0


K = max(1, args.small_concurrency)
workers = [W(k, g) for k, g in enumerate(gpus)]  # wid 0..G-1: primaries (GPU index k)
workers += [
    W(len(gpus) * s + k, g, s) for s in range(1, K) for k, g in enumerate(gpus)
]  # wid G*s + k: slot s of GPU k (spawned on demand)


def small(b) -> bool:
    return b[0] <= args.small_max_tokens


def primary(w):
    return workers[w.wid % len(gpus)]


def extras(w):
    """The small-only slots of w's GPU."""
    k = w.wid % len(gpus)
    return [workers[len(gpus) * s + k] for s in range(1, K)]


def restricted(w) -> bool:
    """Small targets only: a slot > 0, or a primary while any slot of its GPU is alive (a GPU on a large target runs one worker)."""
    return w.slot > 0 or any(x.alive for x in extras(w))


# ---------------------------------------------------------------- CUDA MPS (small concurrency): one daemon for the job, node-local pipe dir
mps_dir = None
if (K > 1 and args.mps != "off") or args.mps == "on":
    mps_dir = f"/tmp/fbmps_{os.getpid()}_{secrets.token_hex(4)}"
    os.makedirs(f"{mps_dir}/pipe")
    os.makedirs(f"{mps_dir}/log")
    _menv = dict(
        os.environ,
        CUDA_MPS_PIPE_DIRECTORY=f"{mps_dir}/pipe",
        CUDA_MPS_LOG_DIRECTORY=f"{mps_dir}/log",
    )
    try:
        r = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            check=False,
            env=_menv,
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = r.returncode == 0
        why = (r.stdout + r.stderr).strip()[:200]
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, why = False, str(exc)[:200]
    if ok:
        os.environ.update(
            CUDA_MPS_PIPE_DIRECTORY=f"{mps_dir}/pipe",
            CUDA_MPS_LOG_DIRECTORY=f"{mps_dir}/log",
        )  # every worker is an MPS client
    else:
        mps_dir = None
    ev(
        "mps",
        on=int(ok),
        dir=os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "-"),
        msg=f"'{why}'" if not ok else "-",
    )
if K > 1:
    ev(
        "slots",
        small_concurrency=K,
        small_max_tokens=args.small_max_tokens,
        small_targets=sum(small(it["bucket"]) for it in items),
        gpus=len(gpus),
    )


def stop_mps():
    global mps_dir
    if mps_dir is None:
        return
    d, mps_dir = mps_dir, None
    try:
        subprocess.run(
            ["nvidia-cuda-mps-control"],
            check=False,
            input="quit\n",
            text=True,
            env=dict(os.environ),
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    time.sleep(1.0)
    try:
        import shutil

        (out / "workers" / "mps_control.log").write_bytes(
            Path(f"{d}/log/control.log").read_bytes()
        )
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _cleanup():
    """Whatever ends the parent (an exception, SIGTERM from scancel): the workers' process groups go with it (and the MPS daemon)."""
    for w in workers:
        if w.popen and w.popen.poll() is None:
            try:
                os.killpg(w.popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    stop_mps()


import atexit

atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
done: set = set()
failed: set = set()
fails: dict = defaultdict(int)
nog_targets: set = set()
nog_buckets: set = set()
group_cap: dict = {}
pending_revoke: dict = {}  # t -> requesting worker
rows: dict = {}  # t -> per-target record
worker_py = str(HERE / "fb_multi_worker.py")


def spawn(w):
    w.inc += 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=w.gpu)
    logf = open(out / "workers" / f"w{w.wid}_{w.inc}.log", "ab")
    cmd = [
        sys.executable,
        worker_py,
        "--wid",
        str(w.wid),
        "--addr",
        addr,
        "--authkey-file",
        str(akf),
        "--model",
        args.model,
        "--specs",
        args.specs,
        "--out",
        str(out),
        *worker_args,
    ]
    w.popen = subprocess.Popen(
        cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True
    )
    logf.close()
    w.conn, w.alive, w.ready = None, True, False
    w.spawn_t = w.last_msg = time.time()
    w.queue, w.current, w.computed, w.revoke_pending, w.retiring = (
        [],
        None,
        set(),
        False,
        False,
    )
    ev(
        "spawn",
        w=w.wid,
        gpu=w.gpu,
        inc=w.inc,
        pid=w.popen.pid,
        **({"slot": w.slot} if K > 1 else {}),
    )


def send(w, *msg):
    try:
        w.conn.send(msg)
        return True
    except (OSError, ValueError, AttributeError):
        return False


def assign(w, t, how):
    it = info[t]
    nog = t in nog_targets or it["bucket"] in nog_buckets
    w.queue.append(t)
    send(
        w,
        "assign",
        {"t": t, "bucket": it["bucket"], "tokens": it["tokens"], "nographs": nog},
    )
    rows.setdefault(t, {})["assigned"] = time.time()
    ev(
        "assign",
        w=w.wid,
        t=t,
        how=how,
        bucket=f"{it['bucket'][0]}/{it['bucket'][1]}",
        tokens=it["tokens"],
        est_s=it["cost"],
        nographs=int(nog),
        queued=len(w.queue),
        left=sched.unassigned(),
    )
    if how == "steal":
        ev(
            "steal",
            w=w.wid,
            t=t,
            bucket=f"{it['bucket'][0]}/{it['bucket'][1]}",
            holders=",".join(
                str(x) for x in sorted(sched.holders[it["bucket"]] - {w.wid})
            ),
        )


def _small_t(t) -> bool:
    return small(info[t]["bucket"])


def maybe_spawn_extras(p):
    """Start the small-only slots of primary p's GPU once p runs / holds only small targets and enough small work is unassigned."""
    if (
        K == 1
        or not (p.alive and p.conn)
        or any(x.retiring and x.alive for x in extras(p))
    ):
        return
    if (p.current and not _small_t(p.current[0])) or not all(
        _small_t(t) for t in p.queue
    ):
        return
    if (
        sum(it["cost"] for b, q in sched.queues.items() if small(b) for it in q)
        / len(gpus)
        < args.small_min_left_s
    ):
        return
    if any(
        q and not small(b) and not (sched.holders[b] - {p.wid})
        for b, q in sched.queues.items()
    ):
        return  # an unclaimed large bucket is left: big work first, this primary may still claim it alone
    for x in extras(p):
        if not x.alive and x.respawns < args.max_respawns:
            if x.inc and x.crashed:
                x.respawns += 1
            spawn(x)


def retire_extras(p, why):
    """No small target is left to assign but a large one is: the small-only slots of primary p's GPU run what they hold (still concurrently, and
    a dry worker may revoke it as usual) and exit; the primary takes the large target alone once they are gone (restricted() until then)."""
    for x in extras(p):
        if not x.alive or x.retiring:
            continue
        x.retiring = True
        send(x, "exit")
        ev(
            "retire",
            w=x.wid,
            gpu=x.gpu,
            why=why,
            queued=",".join(x.queue) or "-",
            running=x.current[0] if x.current else "-",
        )


def fill(w):
    if not (w.alive and w.conn) or w.retiring:
        return
    allow = small if restricted(w) else None
    cap = args.lookahead + (0 if w.current else 1)
    while len(w.queue) < cap:
        r = sched.pick(w.wid, allow)
        if r is None:
            break
        assign(w, r[0]["t"], r[1])
    if K > 1 and w.slot == 0:
        large_queued = not w.queue and any(
            not _small_t(t) for v in workers if v is not w and v.alive for t in v.queue
        )
        if (
            allow is not None
            and sched.unassigned(small) == 0
            and (sched.unassigned() > 0 or large_queued)
        ):
            retire_extras(
                w, "large target next"
            )  # the primary takes it alone once the slots have exited
        else:
            maybe_spawn_extras(w)
    if (
        not w.current and not w.queue and not w.revoke_pending
    ):  # dry: take a queued target of a busy worker
        cand = {
            v: [t for t in v.queue if allow is None or _small_t(t)]
            for v in workers
            if v is not w and v.alive and v.queue and v.current
        }
        cand = {v: ts for v, ts in cand.items() if ts}
        if cand:
            v = max(
                cand,
                key=lambda v: (len(cand[v]), sum(info[t]["cost"] for t in cand[v])),
            )
            t = cand[v][-1]
            if t not in pending_revoke and send(v, "revoke", t):
                pending_revoke[t] = w
                w.revoke_pending = True
                ev("revoke?", w=v.wid, t=t, for_w=w.wid)


def mark_failed(t, kind, msg):
    failed.add(t)
    d = out / t / "failed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "marker").write_text(kind + "\n")
    with open(out / "failures.tsv", "a") as f:
        f.write(f"{t}\t{kind}\t{msg[:300]}\n")
    ev("failed", t=t, kind=kind, fails=fails[t])


def target_failed(w, t, kind, msg):
    fails[t] += 1
    if kind == "oom_graphs":
        nog_targets.add(t)
        nog_buckets.add(info[t]["bucket"])
    if fails[t] >= args.max_fails:
        mark_failed(t, kind, msg)
    else:
        sched.requeue(t)
        ev(
            "retry",
            w=w.wid if w else -1,
            t=t,
            kind=kind,
            fails=fails[t],
            nographs=int(t in nog_targets),
        )


def kill(w, reason):
    if not w.alive:
        return
    w.alive = False
    try:
        os.killpg(w.popen.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        w.popen.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    lost = []
    retired = (
        w.retiring and not w.current and not w.computed
    )  # a small-only slot that exited when asked, everything written
    w.crashed = not retired
    if w.current:
        t = w.current[0]
        target_failed(
            w,
            t,
            f"worker_died:{reason}",
            f"worker {w.wid} died ({reason}) while running it",
        )
        lost.append(t)
    for t in reversed(
        sorted(w.computed) + w.queue
    ):  # computed but not yet written, or not started: no penalty
        sched.requeue(t)
        lost.append(t)
    for t, req in list(pending_revoke.items()):
        if req is w or t in w.queue:
            pending_revoke.pop(t)
            if req is not w and req is not None:
                req.revoke_pending = False
    sched.drop_worker(w.wid)
    w.queue, w.current, w.computed, w.revoke_pending = [], None, set(), False
    ev(
        "retired" if retired else "dead",
        w=w.wid,
        inc=w.inc,
        reason=reason,
        rc=w.popen.returncode,
        requeued=",".join(lost) or "-",
    )
    if w.conn:
        try:
            w.conn.close()
        except OSError:
            pass
    w.conn = None
    if (
        w.slot == 0 and sched.unassigned() and w.respawns < args.max_respawns
    ):  # a small-only slot comes back via maybe_spawn_extras
        w.respawns += 1
        spawn(w)
    for v in workers:
        fill(v)


def handle(w, msg):
    ts, kind, *a = msg
    w.last_msg = time.time()
    if kind == "ready":
        w.ready, w.load_s = True, w.load_s + a[0]
        w.first_ready = w.first_ready or ts
        ev("ready", w=w.wid, inc=w.inc, model_load_s=a[0], msa_depth=a[1])
    elif kind == "start":
        t = a[0]
        if t in w.queue:
            w.queue.remove(t)
        w.current = (t, ts)
        r = rows.setdefault(t, {})
        r.update(start=ts, w=w.wid, graphs=a[1])
        ev(
            "start",
            w=w.wid,
            t=t,
            graphs=int(a[1]),
            idle_before_s=round(ts - w.last_finish, 2) if w.last_finish else "-",
        )
        fill(w)
    elif kind == "progress":
        pass
    elif kind == "release":
        ev("graphs", w=w.wid, t=a[0], released_for_res_bucket=a[1])
    elif kind == "computed":
        t, gpu_s, lw = a
        w.current = None
        w.computed.add(t)
        dt = ts - rows[t]["start"]
        w.busy_s += dt
        w.last_finish = ts
        rows[t].update(computed=ts, gpu_s=gpu_s, load_wait_s=lw)
        ev("computed", w=w.wid, t=t, gpu_s=gpu_s, load_wait_s=lw, est_s=info[t]["cost"])
        fill(w)
    elif kind == "done":
        t = a[0]
        w.computed.discard(t)
        done.add(t)
        w.n_done += 1
        sched.remove(t)
        rows.setdefault(t, {})["done"] = ts
        ev("done", w=w.wid, t=t, n_done=len(done), of=len(items))
    elif kind == "fail":
        t, fk, m = a
        if w.current and w.current[0] == t:
            w.busy_s += ts - w.current[1]
            w.last_finish = ts
            w.current = None
        w.computed.discard(t)
        ev("fail", w=w.wid, t=t, kind=fk, msg=f"'{m[:120]}'")
        target_failed(w, t, fk, m)
        fill(w)
    elif kind == "revoked":
        t, ok = a
        req = pending_revoke.pop(t, None)
        if req is not None:
            req.revoke_pending = False
        ev("revoked", w=w.wid, t=t, ok=int(ok), for_w=req.wid if req else -1)
        if ok:
            if t in w.queue:
                w.queue.remove(t)
            if req is not None and req.alive and req.conn and not req.queue:
                sched.adopt(req.wid, t)
                assign(req, t, "revoked")
            else:
                sched.requeue(t)
        for v in workers:
            fill(v)
    elif kind == "groupcap":
        group_cap[int(a[0])] = int(a[1])
        ev("groupcap", w=w.wid, res_bucket=a[0], cap=a[1])
    elif kind == "bye":
        ev("bye", w=w.wid, worker_busy_s=a[0], rc=a[1])


for w in workers:
    if w.slot == 0:  # small-only slots start on demand (maybe_spawn_extras)
        spawn(w)

kill_at = None
if args.kill_test:
    s_, k_ = args.kill_test.split(":")
    kill_at, kill_w = T0 + float(s_), int(k_)


def resolved():
    return len(done | failed) >= len(items)


exiting = False
while True:
    while not incoming.empty():  # a new worker connection: hello identifies it
        c = incoming.get()
        try:
            ts, kind, wid, pid = c.recv()
        except Exception:
            c.close()
            continue
        w = workers[wid]
        if not w.alive or w.popen.pid != pid:
            c.close()
            continue
        w.conn = c
        w.last_msg = time.time()
        send(w, "init", {"group_cap": group_cap})
        ev("hello", w=wid, inc=w.inc, pid=pid)
        fill(w)
    conns = {w.conn: w for w in workers if w.alive and w.conn}
    for c in wait(list(conns), timeout=1.0) if conns else (time.sleep(1.0) or []):
        w = conns[c]
        try:
            while w.alive and c.poll():
                handle(w, c.recv())
        except (EOFError, OSError):
            if exiting:
                w.alive = False
            else:
                kill(w, "connection closed")
    now = time.time()
    for w in workers:
        if not w.alive:
            continue
        if w.popen.poll() is not None and not exiting:
            if w.conn:  # read what it said before it died
                try:
                    while w.conn.poll():
                        handle(w, w.conn.recv())
                except (EOFError, OSError):
                    pass
            kill(w, f"exit {w.popen.returncode}")
        elif w.current and now - w.last_msg > args.stall:
            kill(w, f"stall {now - w.last_msg:.0f}s")
        elif not w.ready and now - w.spawn_t > args.startup_timeout:
            kill(w, "startup timeout")
        elif w.ready and not w.current:
            fill(w)
    if kill_at and now > kill_at:
        kv = workers[kill_w]
        if kv.alive and kv.current:
            ev(
                "killtest",
                w=kill_w,
                pid=kv.popen.pid,
                t=kv.current[0],
                queued=",".join(kv.queue),
            )
            os.kill(kv.popen.pid, signal.SIGKILL)
            kill_at = None
    if resolved() and not exiting:
        exiting = True
        t_end = time.time()
        ev("resolved", done=len(done), failed=len(failed))
        for w in workers:
            if w.alive:
                send(w, "exit")
    if exiting:
        if (
            all(not w.alive or w.popen.poll() is not None for w in workers)
            or time.time() - t_end > 600
        ):
            break
    if not exiting and not any(w.alive for w in workers):
        ev(
            "abort",
            msg="no worker left (respawn limit)",
            done=len(done),
            failed=len(failed),
            left=len(items) - len(done | failed),
        )
        t_end = time.time()
        break

for w in workers:
    if w.popen and w.popen.poll() is None:
        try:
            os.killpg(w.popen.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
stop_mps()
listener.close()
try:
    os.unlink(addr)
except OSError:
    pass

# ---------------------------------------------------------------- summary
wall = t_end - T0
_gl: dict = {}  # per GPU: the last finish of any of its workers
for w in workers:
    if w.last_finish:
        _gl[w.gpu] = max(_gl.get(w.gpu, 0.0), w.last_finish - T0)
fin = list(_gl.values())
per = []
for w in workers:
    lf = (w.last_finish - T0) if w.last_finish else 0.0
    if not w.inc:  # a small-only slot that never started
        continue
    per.append(
        {
            "worker": w.wid,
            "gpu": w.gpu,
            **({"slot": w.slot} if K > 1 else {}),
            "incarnations": w.inc,
            "targets_done": w.n_done,
            "model_load_s": round(w.load_s, 1),
            "first_ready_s": round(w.first_ready - T0, 1) if w.first_ready else None,
            "busy_s": round(w.busy_s, 1),
            "idle_s": round(
                max(
                    0.0, lf - (w.first_ready - T0 if w.first_ready else 0.0) - w.busy_s
                ),
                1,
            )
            if w.first_ready
            else None,
            "last_finish_s": round(lf, 1),
            "tail_idle_s": round(wall - lf, 1),
        }
    )
summary = {
    "wall_s": round(wall, 1),
    "targets": len(items),
    "done": len(done),
    "failed": sorted(failed),
    "finish_spread_s": round(max(fin) - min(fin), 1) if fin else None,
    "per_gpu": per,
    "order": args.order,
    "lookahead": args.lookahead,
    "alloc_conf": ALLOC,
    "worker_args": worker_args,
    **(
        {
            "small_concurrency": K,
            "small_max_tokens": args.small_max_tokens,
            "mps": os.environ.get("CUDA_MPS_PIPE_DIRECTORY") is not None,
        }
        if K > 1
        else {}
    ),
}
(out / "multi_summary.json").write_text(json.dumps(summary, indent=1))
with open(out / "multi_targets.tsv", "w") as f:
    f.write(
        "target\tworker\ttokens\tbucket\test_s\tstart_s\tcomputed_s\tgpu_s\tload_wait_s\tgraphs\tfails\n"
    )
    for t in [it["t"] for it in items]:
        r = rows.get(t, {})
        st = r.get("start")
        f.write(
            f"{t}\t{r.get('w', '')}\t{info[t]['tokens']}\t{info[t]['bucket'][0]}/{info[t]['bucket'][1]}\t{info[t]['cost']}\t"
            f"{round(st - T0, 1) if st else ''}\t{round(r['computed'] - T0, 1) if r.get('computed') else ''}\t{r.get('gpu_s', '')}\t{r.get('load_wait_s', '')}\t"
            f"{int(r['graphs']) if 'graphs' in r else ''}\t{fails.get(t, 0)}\n"
        )
ev(
    "summary",
    wall_s=summary["wall_s"],
    done=len(done),
    failed=len(failed),
    spread_s=summary["finish_spread_s"],
)
for r in per:
    ev("gpu", **r)
sys.exit(0 if len(done) + len(failed) >= len(items) else 1)
