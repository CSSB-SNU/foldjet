"""Add `engine_label` and `notes` columns to the B200 CSVs (stdlib only).   python b200_label_csv.py <csv dir root holding F3 F4 F6 F9>
Engine labels (user / coordinator 2026-11-22):
  A = F6 engine arm and F4 OpenDDE engine arm: 'engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)'
  B = every other engine arm: 'engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)'
  torch / cuEquivariance arms: no engine kernels ('PyTorch operators' / 'cuEquivariance')."""
import csv, sys
from pathlib import Path

A = "engine main f06154bc7 (3.0.8 + b200 clamp fix + b200 holes fix, unreleased)"
B = "engine 3.0.7 source + b200 TriAttn exp-clamp fix (3698ba9ac, unreleased)"
LN = "LayerNorm w64 rows served by Triton (designed B200 exception: the CUDA route refuses, Triton is faster)"
PER_SIZE = {"boltz2", "chai1", "rosettafold3"}


def rewrite(path, fn):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    cols = list(rows[0])
    for c in ("engine_label", "notes"):
        if c not in cols:
            cols.append(c)
    for r in rows:
        r["engine_label"], r["notes"] = fn(r)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(path.name, len(rows), "rows")


def f3(r):
    n = LN
    if r["model"] in PER_SIZE:
        n += "; measured with one process per size (one process over several sizes gives non-finite output on the B200 on a warm CUDA-graph replay: unresolved)"
    return B, n


def f4_times(r):
    lab = A if r["model"] == "opendde" else B
    return f"engine: {lab}; torch: PyTorch operators", (LN if r["model"] != "opendde" else LN + "; TokenDiTProtenix extra_pair_bias on cuda (holes fix)")


def f4_fail(r):
    if r["arm"] == "engine":
        return (A if r["model"] == "opendde" else B), "ESMFold2 on nucleic-acid-only targets: no protein token (same as A100)"
    return "PyTorch operators", ("out of memory" if str(r.get("oom")) == "True" else "ESMFold2 on nucleic-acid-only targets: no protein token (same as A100)")


def f4_backend(r):
    return (A if r["model"] == "opendde" else B), "engine_cuda_only = no when LayerNorm w64 (designed exception, 8 Triton calls) is served by Triton"


def f6(r):
    arm = r["arm"]
    if arm == "miniworld":
        return A, "engine arm: LayerNorm w64 on Triton (designed exception); fp32 Transition on cuda after the holes fix (no Transition triton)"
    return {"pytorch": "PyTorch operators", "pytorchC": "PyTorch operators + torch.compile", "cueqC": "cuEquivariance + torch.compile"}.get(arm, arm), ""


def f9(r):
    if r.get("arm") == "engine":
        n = ""
        if r.get("predictor") == "chai1":
            n = "n=40: CUDA illegal memory access after near-full reserved memory (176.9/178.35 GiB), classified as OOM (oom_classified_from in the sizes record); reproduction under CUDA_LAUNCH_BLOCKING gives a clean OOM"
        return B, n
    return "PyTorch operators", ""


root = Path(sys.argv[1])
rewrite(root / "F3/F3_foldforge_b200.csv", f3)
rewrite(root / "F4/F4_times_b200.csv", f4_times)
rewrite(root / "F4/F4_failures_b200.csv", f4_fail)
rewrite(root / "F4/F4_backend_engine_b200.csv", f4_backend)
rewrite(root / "F6/f6_runs_b200.csv", f6)
rewrite(root / "F6/f6_summary_b200.csv", f6)
rewrite(root / "F9/F9_b200.csv", f9)
rewrite(root / "F9/F9_b200_sizes.csv", lambda r: (B if r.get("arm") == "engine" else "PyTorch operators", f9({"arm": r.get("arm"), "predictor": r.get("predictor")})[1] if r.get("n") == "40" and r.get("predictor") == "chai1" else ""))
