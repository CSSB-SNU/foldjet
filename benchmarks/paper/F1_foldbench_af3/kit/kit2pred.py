"""Lay out a kit FoldBench run as FoldForge pred/: <pred>/<t>/<t>_seed{s}_sample{k}_pred.cif (symlinks) + ranking.csv (seed,sample,ranking_score).
Per shard: a target re-run in big mode (out_big) takes its big outputs; otherwise the fast outputs (out). Kit ranking_score as written by the kit."""
import json, re, sys
from pathlib import Path

kit, root, pred = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
n_t = n_c = 0
missing = []
for shard in sorted(root.glob("s[0-9]")):
    big = {d.name for d in (shard / "out_big").iterdir() if d.is_dir()} if (shard / "out_big").is_dir() else set()
    targets = sorted({d.name for d in (shard / "out").iterdir() if d.is_dir() and not d.name.startswith(".")} | big)
    if kit == "px1":
        fb = shard / "fallback_targets.txt"
        big_set = set(fb.read_text().split()) if fb.exists() else set()
        targets = sorted(set(targets) | big_set)
    for t in targets:
        if kit == "af3":
            src = shard / ("out_big" if t in big else "out") / t
            rows = []
            for line in (src / f"{t}_ranking_scores.csv").read_text().splitlines()[1:] if (src / f"{t}_ranking_scores.csv").exists() else []:
                s, k, v = line.split(",")
                cif = src / f"seed-{s}_sample-{k}" / f"{t}_seed-{s}_sample-{k}_model.cif"
                if cif.exists():
                    rows.append((int(s), int(k), float(v), cif))
        else:
            src = shard / ("out_big" if t in big_set else "out") / t
            rows = []
            for cif in sorted(src.glob("seed_*/predictions/*_sample_*.cif")):
                m = re.search(r"seed_(\d+)/predictions/.+_sample_(\d+)\.cif$", str(cif))
                s, k = int(m[1]), int(m[2])
                js = cif.parent / f"{t}_summary_confidence_sample_{k}.json"
                rows.append((s, k, float(json.loads(js.read_text())["ranking_score"]), cif))
        if len(rows) != 25:
            missing.append(f"{shard.name}/{t}:{len(rows)}")
        if not rows:
            continue
        out = pred / t
        out.mkdir(parents=True, exist_ok=True)
        with (out / "ranking.csv").open("w") as f:
            f.write("seed,sample,ranking_score\n")
            for s, k, v, cif in rows:
                f.write(f"{s},{k},{v}\n")
                link = out / f"{t}_seed{s}_sample{k}_pred.cif"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(cif.resolve())
                n_c += 1
        n_t += 1
print(f"kit2pred {kit}: targets {n_t} cifs {n_c}; targets without 25 samples: {len(missing)} {missing[:20]}")
