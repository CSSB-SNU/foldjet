"""The 5I28 benchmark, FoldJet side (protocol: docs/guides/BENCHMARK-5I28.md).

One process per model: load the model once, then run n = 1, 2, 4, 8, 12, 16 identical
5I28 chains in ascending order. Every n is one ``Predictor`` call with
``execution.benchmark_repeats`` warm repeats: the first forward of a size is cold, the
median of the warm forwards is the number (``model_seconds_warm_median``; scope: one
complete model forward, CUDA-synchronized, without featurization / loading / output
decoding, see ``foldjet.models.execution.measured_forward``). Each cell is written as
soon as it finishes (``cells/<model>_x<n>.json`` and a line in ``cells.jsonl``), so a
run is resumable and an out-of-memory cell does not lose the earlier ones.

Environment (each defaults to the copy in the repository):
  FOLDFORGE_5I28_FASTA        the 5I28 chain fasta
                              (default: tests/release_references/inputs/5i28/A.fasta)
  FOLDFORGE_5I28_MSA_DB       MSA LMDB with key P0113359
                              (default: benchmarks/5i28_data/a3m_16k.lmdb)
  FOLDFORGE_5I28_TEMPLATE_DB  template LMDB with key P0113359
                              (default: benchmarks/5i28_data/template.lmdb)
Checkpoints come from FOLDFORGE_CHECKPOINT_DIR / the default checkpoint directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

import foldjet as ff

KEY = "P0113359"
REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "FOLDFORGE_5I28_FASTA": REPO / "tests/release_references/inputs/5i28/A.fasta",
    "FOLDFORGE_5I28_MSA_DB": REPO / "benchmarks/5i28_data/a3m_16k.lmdb",
    "FOLDFORGE_5I28_TEMPLATE_DB": REPO / "benchmarks/5i28_data/template.lmdb",
}
TOKENS_PER_CHAIN = 128
SIZES = (1, 2, 4, 8, 12, 16)

#: model -> (recycles, template_n, extra config). Trunk passes are 4 everywhere, as the
#: kits: ``recycles`` counts total trunk passes for every family, AF3 included (records
#: before 2026-11-17 ran AF3 with 3 = 3 passes); ESMFold2's kit runs num_loops 3 = 4 steps. RoseTTAFold3 takes one
#: coordinate set per chain (template_n 1) and its family turns input templates off by
#: default: switched on here, as in the 2026-10-29 run.
MODELS: dict[str, tuple[int, int, dict]] = {
    "protenix1": (4, 4, {}),
    "protenix2": (4, 4, {}),
    "opendde": (4, 4, {}),
    "esmfold2": (4, 0, {}),  # the checkpoint takes no templates
    "esmfold2-fast": (4, 0, {}),  # same family, 24 trunk blocks, no MSA stack; no kit
    "intellifold2": (4, 4, {}),  # the AF3 graph at wider channels; no kit
    "af3": (4, 4, {}),
    "boltz2": (4, 4, {}),
    "chai1": (4, 4, {}),
    "openfold3": (4, 4, {}),
    "rosettafold3": (
        4,
        1,
        {"debug": {"dense_spec_override": {"use_input_templates": True}}},
    ),
}


def letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def make_spec(n: int, template_n: int) -> dict:
    """N identical 5I28 chains, the real MSA and the real templates of key P0113359.

    ``template_dedup`` stays at its default: 4 distinct.
    """
    fasta, msa_db, template_db = (
        os.environ.get(k, str(DEFAULTS[k]))
        for k in (
            "FOLDFORGE_5I28_FASTA",
            "FOLDFORGE_5I28_MSA_DB",
            "FOLDFORGE_5I28_TEMPLATE_DB",
        )
    )
    ids = letters(n)
    spec = {
        "name": "5i28",
        "chain_letters": {str(i): c for i, c in enumerate(ids)},
        "fasta": dict.fromkeys(ids, fasta),
        "n_trunk_samples": 1,
        "n_diffusion_samples": 5,
        "diffusion_batch_size": 5,
        "save_trajectory": False,
        "msa": dict.fromkeys(ids, KEY),
        "msa_db": dict.fromkeys(ids, msa_db),
        "template": {str(i): KEY for i in range(n)},
        "template_db": template_db,
        "template_n": template_n,
    }
    if template_n == 0:
        for key in ("template", "template_db", "template_n"):
            del spec[key]
    return spec


def make_config(recycles: int, extra: dict, repeats: int) -> dict:
    config = {
        "backend": "miniworld",
        # bf16 (the benchmark), af3_default | fp32 for precision checks
        "precision": os.environ.get("FOLDFORGE_5I28_PRECISION", "bf16"),
        "mode": "exact",
        "trunk_seed": 0,
        "diffusion_seed": 0,
        "trunk": {"recycles": recycles},
        "diffusion": {"steps": 200},
        "execution": {
            "benchmark_repeats": repeats,
            "compile": True,
            "cuda_graph": True,
            "bucketing": True,
            "scope": "trunk",
        },
    }
    for key, value in extra.items():
        config.setdefault(key, {}).update(value)
    return config


def oom_of(error: BaseException) -> BaseException | None:
    while error is not None:
        if isinstance(error, torch.OutOfMemoryError):
            return error
        error = error.__cause__ or error.__context__
    return None


def release_size_memory() -> tuple[int, int]:
    """Drop what one size leaves on the GPU before the next.

    That is every captured CUDA graph (each keeps a private memory pool; the sizes run
    once each, ascending, so none is replayed again) and the allocator's cached blocks.
    The compiled code and the weights stay. Returns the reserved bytes before and after.
    """
    from foldjet.models.execution import ExecutedCallable

    try:
        from foldjet.models.sampler_graph import SamplerGraphCache
    except ImportError:  # a FoldJet without the per-shape sampler graph cache
        SamplerGraphCache = ()  # noqa: N806 - class alias fallback

    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved()
    for item in gc.get_objects():
        # type(), not isinstance: a dead weakref proxy raises on isinstance
        if type(item) is ExecutedCallable and item.graphs:
            item.graphs.clear()
            item.last_capture_bytes = 0
        # the sampler graphs cached per shape (R26), with their static folds
        elif (
            SamplerGraphCache
            and type(item) is SamplerGraphCache
            and item.entries
            and not os.environ.get("FF_5I28_KEEP_SAMPLER_CACHE")
        ):
            item.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return before, torch.cuda.memory_reserved()


def git(*args: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed git argv, no untrusted input
            ["git", *args],  # noqa: S607 - git resolved from PATH on purpose
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return ""


def meta(model: str, config: dict) -> dict:
    from importlib import metadata

    import foldjet_engine.modules as engine_modules

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "model": model,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "miniworld_engine": version("foldjet-engine"),
        # the installed version above can differ (PYTHONPATH)
        "miniworld_engine_loaded_from": engine_modules.__file__,
        "miniworld_engine_kernels": version("foldjet-engine-kernels-sm90-cu129"),
        "foldforge_commit": git("rev-parse", "HEAD"),
        "foldjet_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "config": config,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="run directory (cells/, specs/, meta.json)",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="warm repeats per cell (the 2026-10-29 run used 3)",
    )
    parser.add_argument(
        "--retry-fragmented",
        action="store_true",
        help=(
            "only the cells that failed with an out-of-memory error of the "
            "'reserved but unallocated' kind (run with "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: the last fallback)"
        ),
    )
    args = parser.parse_args()

    recycles, template_n, extra = MODELS[args.model]
    config = make_config(recycles, extra, args.repeats)
    cells, specs = args.out / "cells", args.out / "specs"
    cells.mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    (args.out / "meta.json").write_text(json.dumps(meta(args.model, config), indent=1))

    todo = []
    for n in sorted(args.sizes):
        path = cells / f"{args.model}_x{n}.json"
        previous = json.loads(path.read_text()) if path.exists() else None
        if args.retry_fragmented:
            if previous and previous["status"] == "oom" and previous.get("fragmented"):
                todo.append(n)
        # ok and oom cells are final (a fragmented oom waits for --retry-fragmented)
        elif previous is None or previous["status"] == "failed":
            todo.append(n)
    print(
        f"RUN model={args.model} sizes={todo} repeats={args.repeats} "
        f"recycles={recycles}",
        flush=True,
    )
    if not todo:
        return 0

    predictor = ff.load_model(args.model, config=config)
    for n in todo:
        spec_path = specs / f"{args.model}_x{n}.yaml"
        spec_path.write_text(yaml.safe_dump(make_spec(n, template_n), sort_keys=False))
        cell = {
            "model": args.model,
            "n": n,
            "tokens": TOKENS_PER_CHAIN * n,
            "repeats": args.repeats,
            "retry": "expandable_segments" if args.retry_fragmented else None,
        }
        started = time.time()
        try:
            result = predictor(ff.load_inputs(spec_path))
            report = result.report
            cell |= {
                "status": "ok",
                "model_seconds_cold": report["model_seconds_cold"],
                "model_seconds_warm": report["model_seconds_warm"],
                "model_seconds_warm_median": report["model_seconds_warm_median"],
                "trunk_passes": report.get("trunk_passes"),
                "compile": report.get("compile"),
                "cuda_graph": report.get("cuda_graph"),
                "cuda_graph_captures": report.get("cuda_graph_captures"),
                "cuda_graph_evictions": report.get("cuda_graph_evictions"),
                **{k: v for k, v in report.items() if k.startswith("sampler_graph_")},
                "peak_allocated_bytes": report.get("model_peak_allocated_bytes"),
                "peak_reserved_bytes": report.get("model_peak_reserved_bytes"),
            }
        except Exception as error:  # noqa: BLE001 - a failed cell is data, not a crash
            oom = oom_of(error)
            cell |= {
                "status": "oom" if oom else "failed",
                "error": str(oom or error).splitlines()[0][:300],
                "fragmented": bool(
                    oom and "reserved by PyTorch but unallocated" in str(oom)
                ),
            }
            if not oom:
                cell["traceback"] = repr(error)[:600]
            # The exception's traceback holds the failed forward's frames and every
            # tensor they had: drop them, or the next size starts with the
            # out-of-memory size's activations still allocated.
            traceback.clear_frames(error.__traceback__)
            error = oom = None
            result = None
        cell["wall_seconds"] = round(time.time() - started, 1)
        (cells / f"{args.model}_x{n}.json").write_text(json.dumps(cell, indent=1))
        with (args.out / "cells.jsonl").open("a") as handle:
            handle.write(json.dumps(cell) + "\n")
        warm = cell.get("model_seconds_warm_median")
        print(
            f"CELL {args.model} n={n} status={cell['status']} warm_median={warm} "
            f"warm={cell.get('model_seconds_warm')} "
            f"cold={cell.get('model_seconds_cold')} compile={cell.get('compile')} "
            f"wall={cell['wall_seconds']}s",
            flush=True,
        )
        del result
        # each size starts from the weights alone: no pools of the smaller sizes
        # (OOM at n=16)
        before, after = release_size_memory()
        print(
            f"FREE after n={n}: reserved {before / 2**30:.1f} -> "
            f"{after / 2**30:.1f} GiB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
