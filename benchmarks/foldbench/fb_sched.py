"""Bucket-affine work scheduler of fb_multi.py: pure logic (no CUDA, no IO), shared by the parent process and the `--simulate` dry run.

Targets are grouped by their exact FoldJet size bucket (BucketShape.select(tokens, atoms, 1): token bucket, atom bucket). Inside a bucket the targets
are ordered by estimated cost, largest first. A worker asks for its next target with `pick(worker)`:
  (a) affinity: the next target of the bucket the worker holds (its CUDA graphs stay warm);
  (b) claim:    that bucket is empty -> the unclaimed bucket (no live holder) with the largest priority (`order`: "total" = remaining summed cost,
                "largest" = largest single remaining target) -> big work first, the tail is small targets;
  (c) steal:    no unclaimed bucket is left -> share the bucket with the largest remaining cost with its holder(s).
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from pathlib import Path


class Scheduler:
    def __init__(self, items: list[dict], order: str = "total", first=None):
        """items: dicts with t (name), bucket (tuple), cost (est. seconds), tokens, atoms. first(bucket) -> bool: such buckets are claimed / stolen
        before all others whatever their priority (small concurrency: large buckets before the small ones the extra slots can share)."""
        self.first = first
        self.order = order
        self.info = {it["t"]: it for it in items}
        by = defaultdict(list)
        for it in items:
            by[it["bucket"]].append(it)
        self.queues = {
            b: deque(sorted(v, key=lambda x: (-x["cost"], x["t"])))
            for b, v in by.items()
        }
        self.holders: dict = defaultdict(set)  # bucket -> workers currently holding it
        self.wbucket: dict = {}  # worker -> bucket it holds

    # ---- queries
    def remaining_cost(self, b) -> float:
        return sum(x["cost"] for x in self.queues.get(b, ()))

    def _prio(self, b) -> float:
        q = self.queues[b]
        return (
            self.remaining_cost(b)
            if self.order == "total"
            else max(x["cost"] for x in q)
        )

    def unassigned(self, allow=None) -> int:
        return sum(len(q) for b, q in self.queues.items() if allow is None or allow(b))

    # ---- mutations
    def _hold(self, w, b) -> None:
        old = self.wbucket.get(w)
        if old is not None:
            self.holders[old].discard(w)
        self.wbucket[w] = b
        self.holders[b].add(w)

    def pick(self, w, allow=None):
        """-> (item, how) with how in {affinity, claim, steal}, or None when nothing is unassigned (among the buckets `allow(bucket)` accepts,
        when given: a small-concurrency slot only takes small buckets)."""
        b = self.wbucket.get(w)
        if b is not None and self.queues.get(b) and (allow is None or allow(b)):
            return self.queues[b].popleft(), "affinity"
        live = [x for x, q in self.queues.items() if q and (allow is None or allow(x))]
        if not live:
            if b is not None:
                self.holders[b].discard(w)
                self.wbucket.pop(w, None)
            return None
        free = [x for x in live if not (self.holders[x] - {w})]
        how = "claim" if free else "steal"
        best = max(
            free or live,
            key=lambda x: (bool(self.first and self.first(x)), self._prio(x), x),
        )
        self._hold(w, best)
        return self.queues[best].popleft(), how

    def requeue(self, t: str) -> None:
        """Back to the front of its bucket (a dead worker's in-flight target, a retry)."""
        it = self.info[t]
        q = self.queues.setdefault(it["bucket"], deque())
        if it not in q:
            q.appendleft(it)

    def remove(self, t: str) -> None:
        it = self.info[t]
        q = self.queues.get(it["bucket"])
        if q and it in q:
            q.remove(it)

    def drop_worker(self, w) -> None:
        b = self.wbucket.pop(w, None)
        if b is not None:
            self.holders[b].discard(w)

    def adopt(self, w, t: str) -> None:
        """Worker w got target t outside pick() (a revoked lookahead target): it now holds t's bucket."""
        self._hold(w, self.info[t]["bucket"])


# ---- cost model ------------------------------------------------------------------------------------------------------------------------------------


def cost_model(
    per_target_tsv: Path | None,
    run_dirs=(),
    sizes: dict | None = None,
    fit_min_tokens: int = 256,
):
    """est(t, tokens) -> seconds. Measured per-target seconds when known: `run_dirs` (earlier FoldBench outputs, `<dir>/<t>/run.json` total_s,
    read-only) first, then `per_target_tsv` (target, af3_tokens, seconds; foldbench_cost_s.tsv: the measured FoldJet seconds per target of the
    2026-11-16/17 FoldBench runs). Otherwise a log-log least-squares fit
    sec = a * tokens^b over the measured targets with >= `fit_min_tokens` tokens (the unmeasured ones are the large ones); tokens^2.5 without data."""
    import json

    measured: dict[str, float] = {}
    toks: dict[str, float] = {}
    if per_target_tsv is not None and Path(per_target_tsv).exists():
        for line in Path(per_target_tsv).read_text().splitlines()[1:]:
            f = line.split("\t")
            try:
                tok, sec = float(f[1]), float(f[2])
            except (IndexError, ValueError):
                continue
            if tok > 0 and sec > 0:
                measured[f[0]], toks[f[0]] = sec, tok
    for d in run_dirs:
        for rj in Path(d).glob("*/run.json"):
            t = rj.parent.name
            try:
                sec = float(json.loads(rj.read_text())["total_s"])
            except Exception:  # a partial file
                continue
            if sec > 0 and (rj.parent / "done").exists():
                measured[t] = sec
                if sizes and t in sizes:
                    toks[t] = float(sizes[t])
    pts = [
        (math.log(toks[t]), math.log(s))
        for t, s in measured.items()
        if toks.get(t, 0) >= fit_min_tokens
    ]
    if len(pts) >= 10:
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sum(
            (p[0] - mx) ** 2 for p in pts
        )
        a = math.exp(my - b * mx)
    else:
        a, b = 2.0 / 128**2.5, 2.5
    desc = f"measured for {len(measured)} targets, else {a:.3g} * tokens^{b:.2f}"

    def est(t: str, tokens: int) -> float:
        return measured.get(t, a * max(tokens, 1) ** b)

    return est, desc


# ---- dry run ---------------------------------------------------------------------------------------------------------------------------------------


def simulate(
    items: list[dict], n_workers: int, order: str, switch_s: float = 0.0
) -> dict:
    """Event simulation with the estimated costs (no lookahead, no failures): makespan, per-worker busy, finish spread, picks by kind."""
    s = Scheduler(items, order)
    clock = [0.0] * n_workers
    last_b = [None] * n_workers
    hows: dict = defaultdict(int)
    busy = [0.0] * n_workers
    active = set(range(n_workers))
    while active:
        w = min(active, key=lambda k: clock[k])
        r = s.pick(w)
        if r is None:
            active.discard(w)
            continue
        it, how = r
        hows[how] += 1
        dt = it["cost"] + (switch_s if it["bucket"] != last_b[w] else 0.0)
        last_b[w] = it["bucket"]
        clock[w] += dt
        busy[w] += dt
    return {
        "order": order,
        "makespan_s": max(clock),
        "finish_s": clock,
        "spread_s": max(clock) - min(clock),
        "busy_s": busy,
        "picks": dict(hows),
        "ideal_s": sum(busy) / n_workers,
    }
