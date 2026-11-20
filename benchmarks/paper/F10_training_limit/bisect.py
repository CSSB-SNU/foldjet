"""F10 search driver: finds the longest crop (part a) or the most trunk Pairformer blocks (parts b384 / b768) that train, by an exponential
walk up a fixed ladder and then a bisect; every trial is ONE process (trial.sh). An out-of-memory trial is a recorded result: it is never retried
with another shape. Results are appended to results.jsonl (resumable: a finished trial is read back, not run again).

  python bisect.py --part a|b384|b768 --arm miniworld|pytorch|pytorchC|cueq|cueqC --out DIR [--card h100] [--chunks 0,24,12,6,3,1]

part a   : activation checkpointing ON (policy all), 48 trunk blocks, crop length searched (atoms = 32/3 x tokens, 4,096 at 384, 8,192 at 768),
           augmentation chunk 1 (the smallest memory) for the search; at the longest length that trains, the chunk ladder is run from large to small
           and the first chunk that trains is recorded (the setting "as needed").
part b384: no checkpointing, 384 tokens / 4,096 atoms, trunk Pairformer blocks searched (native 48), chunk 1 in the search, ladder at the maximum.
part b768: the same at 768 tokens / 8,192 atoms.
Standard library only.
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LENGTH_LADDER = [384, 512, 640, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192]
BLOCK_LADDER = [48, 64, 96, 128, 192, 256, 384]


def atoms_for(tokens: int) -> int:
    """Flat atoms of a crop: the recipe ratio 4,096 / 384 (= 32/3), rounded up to the 32-atom query window."""
    return math.ceil(tokens * 32 / 3 / 32) * 32


def parse_log(path: Path) -> dict:
    """Read one trial log into a result row (status ok | oom | nonfinite | error | timeout)."""
    text = path.read_text(errors="replace") if path.exists() else ""
    row: dict = {"status": "error", "error": "no output"}
    bench = next((json.loads(line[6:]) for line in text.splitlines() if line.startswith("BENCH ")), None)
    peak = re.findall(r"^PF_PEAK blocks=\S+ alloc_gib=([\d.]+) reserved_gib=([\d.]+)", text, re.M)
    if peak:
        row["peak_alloc_gib"], row["peak_reserved_gib"] = float(peak[-1][0]), float(peak[-1][1])
    meta = re.search(r"^TRIALMETA (.*)$", text, re.M)
    if meta:
        row.update(dict(kv.split("=", 1) for kv in meta.group(1).split() if "=" in kv))
    oom = re.search(r"^OOM (.*)$", text, re.M)
    err = re.search(r"^ERROR (.*)$", text, re.M)
    if oom:
        row.update(status="oom", error=oom.group(1)[:160])
    elif err:
        row.update(status="error", error=err.group(1)[:300])
    elif bench is not None:
        ok = bench.get("nonfinite_steps", 1) == 0
        row.update(
            status="ok" if ok else "nonfinite",
            error="",
            median_step_s=bench.get("median_s"),
            first_step_s=bench.get("first_step_s"),
            peak_alloc_gib=bench.get("peak_allocated_gib", row.get("peak_alloc_gib")),
            peak_reserved_gib=bench.get("peak_reserved_gib", row.get("peak_reserved_gib")),
            commit=bench.get("commit"),
            engine_version=bench.get("engine_version"),
            engine_file=bench.get("engine_file"),
            gpu=bench.get("gpu"),
            engine_table=bench.get("engine_table"),
        )
    elif "out of memory" in text.lower():
        row.update(status="oom", error="out of memory (no OOM line)")
    return row


class Search:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.results = self.out / "results.jsonl"
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.done: dict[tuple, dict] = {}
        if self.results.exists():
            for line in self.results.read_text().splitlines():
                r = json.loads(line)
                if r["status"] in ("ok", "oom", "nonfinite"):
                    self.done[self.key(r)] = r

    @staticmethod
    def key(r) -> tuple:
        return (r["part"], r["arm"], int(r["tokens"]), int(r["chunk"]), int(r["ckpt"]), int(r["blocks"]))

    def trial(self, tokens: int, chunk: int, ckpt: int, blocks: int) -> dict:
        a = self.args
        key = (a.part, a.arm, tokens, chunk, ckpt, blocks)
        if key in self.done:
            return self.done[key]
        log = self.logs / f"{a.part}_{a.arm}_L{tokens}_c{chunk}_k{ckpt}_n{blocks}.log"
        t0 = time.time()
        try:
            subprocess.run(  # noqa: S603 - fixed script, integer arguments
                [str(HERE / "trial.sh"), a.arm, str(tokens), str(atoms_for(tokens)), str(chunk), str(ckpt), str(blocks), str(log)],
                timeout=a.trial_timeout,
                check=False,
                env={**os.environ, "OUT": str(self.out)},
            )
            row = parse_log(log)
        except subprocess.TimeoutExpired:
            row = {"status": "timeout", "error": f"timeout {a.trial_timeout}s"}
        row.update(
            part=a.part, arm=a.arm, card=a.card, tokens=tokens, atoms=atoms_for(tokens), chunk=chunk, ckpt=ckpt, blocks=blocks,
            wall_s=round(time.time() - t0, 1), log=str(log), slurm_job=os.environ.get("SLURM_JOB_ID", "none"),
        )
        with self.results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"TRIAL {a.part} {a.arm} tokens={tokens} chunk={chunk} ckpt={ckpt} blocks={blocks} -> {row['status']} "
              f"peak={row.get('peak_alloc_gib')} {row.get('error', '')[:80]}", flush=True)
        if row["status"] in ("ok", "oom", "nonfinite"):
            self.done[key] = row
        return row

    def search(self, ladder: list[int], test, step: int = 1) -> tuple[int, int | None, str]:
        """Return (largest value that trains (0 if none), first value that fails or None, note). ``test(v)`` -> status."""
        lo, hi = 0, None
        for v in ladder:
            s = test(v)
            if s == "ok":
                lo = v
            elif s in ("oom", "nonfinite"):
                hi = v
                break
            else:
                return lo, v, f"stopped: non-OOM failure ({s}) at {v}"
        if hi is None:
            return lo, None, "reached the top of the ladder"
        while hi - lo > step:  # bisect on the grid
            mid = (lo + hi) // 2
            mid = max(step, (mid // step) * step) if step > 1 else mid
            if mid <= lo or mid >= hi:
                break
            s = test(mid)
            if s == "ok":
                lo = mid
            elif s in ("oom", "nonfinite"):
                hi = mid
            else:
                return lo, mid, f"stopped: non-OOM failure ({s}) at {mid}"
        return lo, hi, "bisect done"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=("a", "b384", "b768"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default=os.environ.get("CARD", "h100"))
    ap.add_argument("--chunks", default="0,24,12,6,3,1", help="chunk ladder at the maximum, large to small; the LAST value is the one of the search")
    ap.add_argument("--grid", type=int, default=64, help="part a: token grid of the bisect")
    ap.add_argument("--trial-timeout", type=int, default=2400)
    ap.add_argument("--length-ladder", default=None)
    ap.add_argument("--block-ladder", default=None)
    args = ap.parse_args()
    s = Search(args)
    chunks = [int(c) for c in args.chunks.split(",")]
    search_chunk = chunks[-1]
    if args.part == "a":
        ladder = [int(x) for x in args.length_ladder.split(",")] if args.length_ladder else LENGTH_LADDER
        lo, hi, note = s.search(ladder, lambda L: s.trial(L, search_chunk, 1, 0)["status"], step=args.grid)
        point = {"tokens": lo, "blocks": 0, "ckpt": 1}
    else:
        tokens = int(args.part[1:])
        ladder = [int(x) for x in args.block_ladder.split(",")] if args.block_ladder else BLOCK_LADDER
        lo, hi, note = s.search(ladder, lambda n: s.trial(tokens, search_chunk, 0, n)["status"], step=1)
        point = {"tokens": tokens, "blocks": lo, "ckpt": 0}
    summary = {"part": args.part, "arm": args.arm, "card": args.card, "max": lo, "first_failing": hi, "note": note, "chunk_of_search": search_chunk}
    if lo > 0:  # the chunk "as needed" at the maximum: the largest chunk that trains
        at_max = None
        for c in chunks:
            r = s.trial(point["tokens"] if args.part == "a" else point["tokens"], c, point["ckpt"], lo if args.part != "a" else 0)
            if r["status"] == "ok":
                at_max = {"chunk": c, "peak_alloc_gib": r.get("peak_alloc_gib"), "peak_reserved_gib": r.get("peak_reserved_gib"), "median_step_s": r.get("median_step_s")}
                break
        summary["at_max"] = at_max
    (s.out / f"summary_{args.part}_{args.arm}.json").write_text(json.dumps(summary, indent=1))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
