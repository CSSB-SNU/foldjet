#!/usr/bin/env python
"""Compare two F6 step profiles (H100 vs B200): kernel classes, ops (eager runs), per-micro wall. CPU only (run through srun / on the B200 host, not on a login node).

    python compare_profiles.py OUT_H100 OUT_B200 [TAG]     # TAG e.g. _protenix1_miniworld (graph run) ; TAG_eager has ops*.json too
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "_protenix1_miniworld"
    sa, sb = load(a, f"profile_summary{tag}.json"), load(b, f"profile_summary{tag}.json")
    if sa and sb:
        print(f"step (8 micro): A wall {sa['wall_ms']:.0f} ms busy {sa['gpu_busy_ms']:.0f} idle {sa['gpu_idle_fraction']:.3f} kernels {sa['n_kernels']} | B wall {sb['wall_ms']:.0f} ms busy {sb['gpu_busy_ms']:.0f} idle {sb['gpu_idle_fraction']:.3f} kernels {sb['n_kernels']}")
        print(f"{'class':<62}{'A ms':>10}{'B ms':>10}{'B/A':>7}{'B-A ms':>10}")
        for k in sorted(set(sa["classes"]) | set(sb["classes"]), key=lambda k: -(sb["classes"].get(k, {"ms": 0})["ms"] - sa["classes"].get(k, {"ms": 0})["ms"])):
            x, y = sa["classes"].get(k, {"ms": 0.0}), sb["classes"].get(k, {"ms": 0.0})
            print(f"{k:<62}{x['ms']:>10.1f}{y['ms']:>10.1f}{y['ms'] / max(x['ms'], 1e-9):>7.2f}{y['ms'] - x['ms']:>10.1f}")
    oa, ob = load(a, f"ops{tag}.json"), load(b, f"ops{tag}.json")
    if oa and ob:
        da, db = {o["op"]: o for o in oa}, {o["op"]: o for o in ob}
        print(f"\n{'op (self device ms over 8 micro steps)':<80}{'A ms':>10}{'B ms':>10}{'B/A':>7}{'B-A ms':>10}")
        names = sorted(set(da) | set(db), key=lambda k: -(db.get(k, {"self_device_ms": 0})["self_device_ms"] - da.get(k, {"self_device_ms": 0})["self_device_ms"]))
        for k in names[:40]:
            x, y = da.get(k, {"self_device_ms": 0.0})["self_device_ms"], db.get(k, {"self_device_ms": 0.0})["self_device_ms"]
            print(f"{k[:79]:<80}{x:>10.1f}{y:>10.1f}{y / max(x, 1e-9):>7.2f}{y - x:>10.1f}")
        print("(op names of an arch-specific kernel differ between cards: an op only on one side shows B/A 0 or huge; compare the op, not the raw kernel)")


if __name__ == "__main__":
    main()
