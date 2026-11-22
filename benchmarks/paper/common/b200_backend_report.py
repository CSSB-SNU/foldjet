"""Per-run backend report of the engine arms of one card: served / refused calls per engine class from the BACKEND_TABLE rows (F4 meta/*_backends.jsonl).
python b200_backend_report.py <F4 root of the card> <out csv>      columns: model,arm,all_cuda,triton_calls,refused_reasons,classes_with_refusals,engine_cuda_only"""
import csv, json, sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for f in sorted((root / "meta").glob("*_backends.jsonl")):
    arm, model = f.name.removesuffix("_backends.jsonl").split("__")
    first = json.loads(f.read_text().splitlines()[0])
    tab = first.get("table") or {}
    tri, reasons, classes = 0, {}, []
    for cls, v in (tab.get("classes") or {}).items():
        t = (v.get("served") or {}).get("triton", 0)
        tri += t
        if v.get("refused"):
            classes.append(cls)
            for r, n in v["refused"].items():
                reasons[r] = reasons.get(r, 0) + n
    rows.append({"model": model, "arm": arm, "all_cuda": tab.get("all_cuda"), "triton_calls": tri, "pytorch_calls": sum((v.get("served") or {}).get("pytorch", 0) for v in (tab.get("classes") or {}).values()),
                 "refused_reasons": json.dumps(reasons), "classes_with_refusals": ";".join(classes),
                 "engine_cuda_only": "yes" if tab.get("all_cuda") else "no"})
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", out)
