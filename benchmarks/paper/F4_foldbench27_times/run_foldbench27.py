"""F4: 11 predictors x the 27 FoldBench targets, time of the model forward with the engine's kernels against PyTorch's operators. One (predictor, arm) per process.

    python run_foldbench27.py MODEL ARM [--out DIR] [--only t1,t2] [--deadline-s N]
    ARM: torch (backend pytorch) | engine (miniworld, bf16 kernels) | engine_fp8 (miniworld + FP8, H100 only) | anthropic (the engine's connection to Anthropic's kernels)

Same inputs and seeds for every arm: FoldBench's own inputs (`$FB_ROOT/inputs/<target>/data.yaml`: real MSA, templates where the checkpoint takes them),
trunk_seed = diffusion_seed = 0, 200 steps, the 5I28 benchmark's 4 trunk passes, 5 samples (the input spec), bf16, mode exact. compile + CUDA graphs (scope trunk,
as F1 / F3) on every arm since 2026-11-24 (FFPAPER_F4_EXEC=eager: the earlier eager protocol), so the arms differ only in the backend; `benchmark_repeats: 1` = one cold and one warm forward; the number is the warm model time
(`model_seconds_warm_median`, CUDA-synchronized, no featurization / loading / writing). The structures of the (last) forward are saved for F5.

Output  <out>/rows/<arm>__<model>.jsonl     one row per target: status ok | error | timeout, seconds, run information
        <out>/pred/<arm>/<model>/<target>/  FoldForge's result files (CIFs, npz with coordinates / plddt / full_pae), chain_names.json
Resumable: a target with a final row is skipped; a target started twice without a result is recorded as `timeout` (it does not fit the job).
Failures (also out of memory) are results: recorded, never retried here. ESMFold2 on RNA / DNA-only targets needs a FoldForge with `allow_no_protein`.
Run through Slurm / the card's GPU lock only (never on a login node).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
import ffpaper  # noqa: E402

_model = sys.argv[1] if len(sys.argv) > 1 else ""
_arm = sys.argv[2] if len(sys.argv) > 2 else ""
os.environ.setdefault("FFPAPER_FP8_GUARD", "lazy")  # lazy guards (no host sync per call); fp8.check() below redoes a saturated forward -- the sync guard cost 0.1-0.35 s / forward (AF3, engine 2026-11-23)
# 2026-11-24: compile + CUDA graphs by default (the way the predictors run; eager made the FP8 arms host-bound, B200 FP8 / bf16 1.10-1.27);
# FFPAPER_F4_EXEC=eager restores the earlier eager protocol
GRAPH = os.environ.get("FFPAPER_F4_EXEC", "graph") != "eager"
if _arm in ffpaper.ARMS:
    ffpaper.apply_arm_env(_arm, _model)

import yaml  # noqa: E402

import foldforge as ff  # noqa: E402
from foldforge import paths  # noqa: E402

FB = Path(os.environ.get("FB_ROOT", Path.home() / "data/foldbench"))
RF3_EXTRA = {"debug": {"dense_spec_override": {"use_input_templates": True}}}
# model -> (trunk passes, extra config, templates). FoldForge's `recycles` = total trunk passes for every family, AF3 included. ESMFold2 takes no templates; RoseTTAFold3
# runs without FoldBench templates (they ruin it: lDDT 0.25-0.68 vs 0.89-0.94 without) and takes ONE coordinate set per chain where templates are on.
MODELS: dict[str, tuple[int, dict, bool]] = {
    "af3": (4, {}, True), "boltz2": (4, {}, True), "chai1": (4, {}, True), "esmfold2": (4, {}, False), "intellifold2": (4, {}, True),
    "opendde": (4, {}, True), "openfold3": (4, {}, True), "openfold3-preview2": (4, {}, True), "protenix1": (4, {}, True), "protenix2": (4, {}, True),
    "rosettafold3": (4, RF3_EXTRA, False),
}
MSA_DB = {"P": FB / "msa/a3m.lmdb", "R": FB / "msa/a3m_rna.lmdb"}  # key prefix -> LMDB; DNA has no MSA
RETRY_ERRORS = ("LMDB key", "Branched FASTA", "No protein token")  # causes fixed in FoldForge after earlier rounds: those targets are tried again
_TEMPLATE_ENV = None


def targets() -> list[tuple[str, str, int]]:
    rows = [line.split("\t") for line in (HERE / "targets27.tsv").read_text().splitlines()[1:]]
    return sorted(((r[2], r[0], int(r[3])) for r in rows), key=lambda t: (t[2], t[0]))


def template_has(key: str) -> bool:
    """Whether the FoldBench template LMDB holds this key (opened once, read-only)."""
    global _TEMPLATE_ENV  # noqa: PLW0603
    import lmdb

    if _TEMPLATE_ENV is None:
        _TEMPLATE_ENV = lmdb.open(str(FB / "template/template.lmdb"), readonly=True, lock=False, subdir=True)
    with _TEMPLATE_ENV.begin() as txn:
        return txn.get(str(key).encode()) is not None


def spec_of(target: str, templates: bool, spec_dir: Path) -> tuple[Path, dict]:
    """FoldBench's data.yaml (a3m_db / a3m_key per chain index) as a FoldForge InferenceSpec (msa_db / msa per chain letter). Returns (yaml path, chain names)."""
    src = yaml.safe_load((FB / "inputs" / target / "data.yaml").read_text())
    letters = src["chain_letters"]
    msa, msa_db = {}, {}
    for index, key in src.get("a3m_key", {}).items():
        db = MSA_DB.get(str(key)[:1])
        if db is not None:
            msa[letters[index]] = key
            msa_db[letters[index]] = str(db)
    spec = {
        "name": src["name"], "chain_letters": letters, "fasta": src["fasta"], "msa": msa, "msa_db": msa_db,
        "n_trunk_samples": src["n_trunk_samples"], "n_diffusion_samples": src["n_diffusion_samples"],
        "diffusion_batch_size": src["diffusion_batch_size"], "save_trajectory": False,
        "allow_unbonded_branched": True,  # FoldBench's AF3 inputs carry glycans as branched chains without explicit bonds
    }
    # released checkpoints take protein templates only; keep a chain's template when its MSA key is a protein key and the template LMDB holds it
    protein_templates = {i: k for i, k in src.get("template", {}).items() if str(src.get("a3m_key", {}).get(i, ""))[:1] == "P" and template_has(k)}
    if templates and protein_templates:
        spec["template_db"] = str(FB / "template/template.lmdb")
        spec["template"] = protein_templates
        spec["template_offset"] = {i: o for i, o in src.get("template_offset", {}).items() if i in protein_templates}
    dst = spec_dir / f"{target}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(spec))
    return dst, src["chain_names"]


def seconds(report: dict) -> dict:
    flat: dict = {}

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}{k}.", v)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and "second" in prefix:
            flat[prefix[:-1]] = value

    walk("", report)
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("arm", choices=list(ffpaper.ARMS))
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("PAPER_OUT", Path.home() / "foldforge-scratch/paper")) / "F4" / os.environ.get("CARD", "card"))
    parser.add_argument("--only", default=os.environ.get("ONLY", ""), help="comma-separated target ids (smoke runs)")
    parser.add_argument("--deadline-s", type=float, default=float("inf"), help="stop starting new targets after this many seconds (resumable)")
    args = parser.parse_args()
    start = time.monotonic()
    recycles, extra, templates = MODELS[args.model]
    only = {t for t in args.only.split(",") if t}
    jsonl = args.out / "rows" / f"{args.arm}__{args.model}.jsonl"
    rows = ffpaper.read_jsonl(jsonl)

    def final(r: dict) -> bool:
        return r["status"] in ("ok", "timeout") or (r["status"] == "error" and not any(s in r.get("error", "") for s in RETRY_ERRORS))

    finished = {r["target"] for r in rows if final(r)}
    started: dict[str, int] = {}
    for r in rows:
        if r["status"] == "started":
            started[r["target"]] = started.get(r["target"], 0) + 1
        elif r["status"] in ("ok", "error", "timeout") and r["target"] in started:
            started[r["target"]] -= 1
    todo = []
    for target, task, tokens in targets():
        if (only and target not in only) or target in finished:
            continue
        if started.get(target, 0) >= 2:
            ffpaper.append_jsonl(jsonl, {"target": target, "task": task, "tokens": tokens, "model": args.model, "arm": args.arm, "status": "timeout",
                                         "note": "started twice without a result"})
            continue
        todo.append((target, task, tokens))
    print(f"RUN F4 model={args.model} arm={args.arm} todo={[t[0] for t in todo]}", flush=True)
    if not todo:
        sys.exit(incomplete(jsonl, only))
    config = {
        "backend": ffpaper.BACKEND[args.arm], "precision": "bf16", "mode": "exact", "trunk_seed": 0, "diffusion_seed": 0,
        "trunk": {"recycles": recycles}, "diffusion": {"steps": 200},
        "execution": {"benchmark_repeats": 1, "compile": GRAPH, "cuda_graph": GRAPH, "bucketing": True, **({"scope": "trunk"} if GRAPH else {})},
    }
    for key, value in extra.items():
        config.setdefault(key, {}).update(value)
    meta = ffpaper.run_meta(experiment="F4", model=args.model, arm=args.arm, config=config)
    (args.out / "meta").mkdir(parents=True, exist_ok=True)
    (args.out / "meta" / f"{args.arm}__{args.model}.json").write_text(json.dumps(meta, indent=1))
    ident = {k: meta[k] for k in ("card", "gpu", "host", "slurm_job_id", "foldforge_commit", "engine_version_installed", "engine_version_loaded", "engine_loaded_from")}
    import torch

    t_load = time.monotonic()
    with ff.load_model(args.model, config=config, device="cuda:0") as model:
        if args.arm != "torch":
            ffpaper.backend_plan()  # first in the log: every engine class must be served by cuda (a Triton / PyTorch route in an engine run is a bug)
        table_done = args.arm == "torch"
        print(f"loaded {args.model} in {time.monotonic() - t_load:.1f} s gpu={meta['gpu']} engine={meta['engine_version_installed']} from {meta['engine_loaded_from']}", flush=True)
        for target, task, tokens in todo:
            if time.monotonic() - start > args.deadline_s:
                print("deadline reached, stopping", flush=True)
                break
            base = {"target": target, "task": task, "tokens": tokens, "model": args.model, "arm": args.arm, **ident}
            ffpaper.append_jsonl(jsonl, {**base, "status": "started"})
            t0 = time.monotonic()
            try:
                spec_path, chain_names = spec_of(target, templates, args.out / "specs" / ("templates" if templates else "no_templates"))
                inputs = ff.load_inputs(spec_path)
                t1 = time.monotonic()
                redo = 0
                if args.arm == "engine_fp8":
                    import miniworld_engine.fp8 as fp8

                    fp8.check()  # clear a flag left from loading / an earlier target
                torch.cuda.reset_peak_memory_stats()
                result = model(inputs)
                if args.arm == "engine_fp8":
                    while fp8.check() and redo < 3:  # guard contract (docs/fp8.md): a saturated static-scale route -> redo the forward
                        redo += 1
                        result = model(inputs)
                wall = time.monotonic() - t1
                out = args.out / "pred" / args.arm / args.model / target
                result.save(out)
                (out / "chain_names.json").write_text(json.dumps(chain_names))
                report = seconds(result.report)
                ffpaper.append_jsonl(jsonl, {**base, "status": "ok", "fp8_redo": redo, "load_inputs_s": round(t1 - t0, 2), "call_wall_s": round(wall, 3),
                                             "model_seconds_warm_median": result.report.get("model_seconds_warm_median"),
                                             "model_seconds_cold": result.report.get("model_seconds_cold"),
                                             "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                             "report_seconds": report, "out": str(out)})
                if not table_done:  # per-class backend table after the first forward
                    tab = ffpaper.backend_table(model)
                    table_done = True
                    ffpaper.append_jsonl(args.out / "meta" / f"{args.arm}__{args.model}_backends.jsonl", {"target": target, "all_cuda": tab["all_cuda"] if tab else None, "table": tab})
                print(f"ok {target} tokens={tokens} warm={result.report.get('model_seconds_warm_median')} wall={wall:.1f}s redo={redo}", flush=True)
            except Exception as error:  # noqa: BLE001
                ffpaper.append_jsonl(jsonl, {**base, "status": "error", "oom": ffpaper.oom_of(error) is not None, "error": f"{type(error).__name__}: {str(error)[:500]}",
                                             "wall_s": round(time.monotonic() - t0, 2)})
                print(f"error {target}: {type(error).__name__}: {str(error)[:200]}", flush=True)
                error = None
                torch.cuda.empty_cache()
    sys.exit(incomplete(jsonl, only))  # non-zero unless every selected target has a valid final row: a driver must never mark a failed run done


def incomplete(jsonl: Path, only: set[str]) -> int:
    """Count of selected targets without a VALID final row (ok, an out-of-memory error = a result, or a recorded timeout); any other error / no row = invalid."""
    latest: dict[str, dict] = {}
    for r in ffpaper.read_jsonl(jsonl):
        latest[r["target"]] = r
    bad = []
    for target, _task, _tokens in targets():
        if only and target not in only:
            continue
        r = latest.get(target)
        if not (r and (r["status"] in ("ok", "timeout") or (r["status"] == "error" and r.get("oom")))):
            bad.append(target)
    print(f"F4 INCOMPLETE {len(bad)}: {bad}" if bad else "F4 COMPLETE", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    main()
