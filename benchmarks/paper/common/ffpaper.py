"""Shared helpers of the paper experiments F3 / F4 / F5 / F9 (inference timing, sample comparison, inference limit).

Import this module BEFORE ``foldjet`` / ``foldjet_engine`` in a runner: ``apply_arm_env`` sets the FP8 switches, which the engine reads when
its FP8 modules are imported.

Arms (``ARMS``): ``torch`` = backend ``pytorch`` (PyTorch operators), ``engine`` = backend ``miniworld`` (bf16 / fp16 engine kernels),
``engine_fp8`` = ``miniworld`` + the opt-in FP8 path (H100 only), ``anthropic`` = the engine's connection to Anthropic's released kernels.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

ARMS = ("torch", "engine", "engine_fp8", "anthropic")
BACKEND = {"torch": "pytorch", "engine": "miniworld", "engine_fp8": "miniworld", "anthropic": "anthropic"}
CARDS = {"h100": "H100", "a100": "A100", "rtx5090": "5090", "b200": "B200"}

#: FoldJet predictors of F3 / F9 (the 5I28 benchmark's eleven) and of F4 / F5 (the FoldBench 27-target set's eleven; the two lists differ:
#: F4 / F5 have openfold3-preview2 and no esmfold2-fast).
PREDICTORS_5I28 = (
    "protenix1", "protenix2", "opendde", "esmfold2", "esmfold2-fast", "intellifold2", "af3", "boltz2", "chai1", "openfold3", "rosettafold3",
)
PREDICTORS_FOLDBENCH = (
    "af3", "boltz2", "chai1", "esmfold2", "intellifold2", "opendde", "openfold3", "openfold3-preview2", "protenix1", "protenix2", "rosettafold3",
)

_FP8_ENV_PREFIXES = ("FOLDJET_ENGINE_",)


def apply_arm_env(arm: str, model: str) -> None:
    """Set (or clear) the FP8 environment for one arm. Call before importing the engine.

    ``engine_fp8``: ``FOLDJET_ENGINE_FP8=safe`` + ``FOLDJET_ENGINE_FP8_MODEL=<model>`` (the engine keeps TriAttn on the fp16 core for a model outside its
    ``TRIATTN_MODELS``, so the variable is set for every model) + lazy guards (no host sync inside a CUDA-graph capture; the saturation check
    ``fp8.check()`` runs once per cell by the runner). Every other arm: no FP8 variable at all.
    """
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    for key in [k for k in os.environ if k.startswith("FOLDJET_ENGINE_") and "FP8" in k]:
        os.environ.pop(key)
    if arm == "engine_fp8":
        os.environ["FOLDJET_ENGINE_FP8"] = "safe"
        os.environ["FOLDJET_ENGINE_FP8_MODEL"] = model
        if os.environ.get("FFPAPER_FP8_GUARD", "lazy") == "lazy":
            for route in ("TRIMUL", "OPM", "PWA", "TRANSITION"):
                os.environ[f"FOLDJET_ENGINE_{route}_FP8_GUARD"] = "lazy"


def random_init_enabled() -> bool:
    """``FF_PAPER_RANDOM_INIT=1``: memory-limit runs (F9) on a machine without the weights (rented 4x RTX 5090, 2026-11-24)."""
    return os.environ.get("FF_PAPER_RANDOM_INIT") == "1"


def install_random_init(seed: int = 0) -> None:
    """Replace FoldJet's checkpoint load by a seeded random initialisation (opt-in, ``FF_PAPER_RANDOM_INIT=1``; no file is read).

    Every predictor family loads through ``foldjet.models.checkpoints.miniworld.load_into`` (``ff.load_model`` -> ``models.loading``): the
    checkpoint path is resolved without requiring the file (a missing file skips the size check), and ``load_into`` fills the parameters of the
    built model instead of a strict load. Scale as F6's profiler (``F6PROF_RANDOM_INIT``): matrices randn / sqrt(fan_in), norm weights / scales 1,
    biases and other vectors 0; the Fourier noise features (buffers the file normally supplies) randn / rand of the module's own ``dim``.
    Peak memory and whether a size fits do not depend on the values; the outputs are meaningless (finiteness is recorded, not required).
    """
    import torch
    from foldjet.models.checkpoints import miniworld

    import tempfile

    skeletons = Path(tempfile.mkdtemp(prefix="ff_paper_random_init_"))

    def resolve_native(model: str) -> Path:
        # an empty skeleton (the API checks that the path exists); its name is no pinned blob's, so the size check skips it; never read.
        # Fixed input towers that live beside the weights (Chai-1's ESM2 `esm/`) are not model parameters: when the model's checkpoint
        # directory exists, its other entries are linked into the skeleton directory (the weights file itself never is).
        from foldjet.models.checkpoints import _roots

        d = skeletons / model
        d.mkdir(exist_ok=True)
        for root in _roots(model):
            if root.is_dir():
                for entry in root.iterdir():
                    if not entry.name.endswith(miniworld.SUFFIX) and not (d / entry.name).exists():
                        (d / entry.name).symlink_to(entry.resolve())
                break
        path = d / f"random_init_{model}{miniworld.SUFFIX}"
        path.touch()
        return path

    def load_into(model, path, *, family):  # noqa: ARG001 - the signature of the replaced loader
        g = torch.Generator().manual_seed(seed)
        fourier = model.get_submodule("diffusion_head.fourier_embeddings")
        fourier.register_buffer("weight", torch.randn(fourier.dim, generator=g))
        fourier.register_buffer("bias", torch.rand(fourier.dim, generator=g))
        with torch.no_grad():
            for pname, p in model.named_parameters():
                if p.ndim >= 2:
                    p.copy_((torch.randn(p.shape, generator=g, dtype=torch.float32) / (p.shape[-1] ** 0.5)).to(p.device, p.dtype))
                elif pname.endswith(("weight", "scale", "gamma")):
                    p.fill_(1.0)
                else:
                    p.zero_()
        print(f"FF_PAPER_RANDOM_INIT: {family} parameters randomly initialised (seed {seed}; no checkpoint read)", flush=True)
        return {"format": "random_init", "format_version": 0, "source": "random", "mapped_records": 0}

    miniworld.resolve_native = resolve_native
    miniworld.load_into = load_into


def card_of(gpu_name: str) -> str:
    low = gpu_name.lower()
    for key, needle in (("h100", "h100"), ("a100", "a100"), ("rtx5090", "5090"), ("b200", "b200")):
        if needle in low:
            return key
    return low.replace(" ", "_")


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603, S607
    except OSError:
        return ""


def run_meta(**extra) -> dict:
    """Commit / engine / GPU / node / Slurm job of this process (every result file carries it)."""
    from importlib import metadata

    import torch

    repo = Path(__file__).resolve().parents[3]
    try:
        import foldjet_engine

        engine_from = str(list(foldjet_engine.__path__)[0])
    except Exception:  # noqa: BLE001
        engine_from = None
    try:
        engine = metadata.version("foldjet-engine")
    except metadata.PackageNotFoundError:
        engine = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    meta = {
        "foldforge_commit": _git(repo, "rev-parse", "HEAD"),
        "foldjet_dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        "engine_version_installed": engine,
        "engine_loaded_from": engine_from,  # differs from the installed one when PYTHONPATH shadows it (a release candidate)
        "engine_version_loaded": engine_source_version(engine_from) or engine,
        "engine_source_override": os.environ.get("PYTHONPATH", "") or None,
        "torch": torch.__version__,
        "gpu": gpu,
        "card": card_of(gpu) if gpu else None,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if gpu else None,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fp8_env": {k: v for k, v in os.environ.items() if k.startswith("FOLDJET_ENGINE_") and "FP8" in k},
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta.update(extra)
    return meta


def steady_median(forwards: list[float], drop: int = 2) -> float | None:
    """Steady-state value of a cell: forward 1 (cold) and forward 2 are dropped, then the median (>= 1 left, else None)."""
    rest = [x for x in forwards if x is not None][drop:]
    return statistics.median(rest) if rest else None


def fp8_check() -> bool | None:
    """True when an FP8 route saturated since the last call (the engine's guard contract), None when FP8 is not in play."""
    if os.environ.get("FOLDJET_ENGINE_FP8") is None:
        return None
    import foldjet_engine.fp8 as fp8

    return bool(fp8.check())


def fp8_status() -> dict | None:
    if os.environ.get("FOLDJET_ENGINE_FP8") is None:
        return None
    import foldjet_engine.fp8 as fp8

    out: dict = {"status": fp8.status()}
    try:
        out["route_stats"] = fp8.route_stats()
    except Exception as error:  # noqa: BLE001
        out["route_stats_error"] = repr(error)[:200]
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def oom_of(error: BaseException | None) -> BaseException | None:
    import torch

    while error is not None:
        if isinstance(error, torch.OutOfMemoryError):
            return error
        error = error.__cause__ or error.__context__
    return None


def backend_plan(device: str = "cuda:0") -> dict | None:
    """Per op class, the backend FoldJet's engine policy chooses on this card (printed first in a real run: a Triton / PyTorch route in an engine run is a bug)."""
    try:
        import torch
        from foldjet.modules.engine.backends import OPS, choose_backend, op_classes

        dev = torch.device(device)
        plan = {op: {c.__name__: choose_backend("miniworld", c, dev)[0] for c in op_classes(op)} for op in OPS}
        print("BACKEND_PLAN " + json.dumps(plan, default=str), flush=True)
        return plan
    except Exception as error:  # noqa: BLE001 - diagnostics never break a run
        print("BACKEND_PLAN unavailable:", repr(error)[:200], flush=True)
        return None


def backend_table(predictor) -> dict | None:
    """What the modules of a loaded predictor were set to and what the engine really served (host calls) per class, after a forward. `all_cuda`: every class set to cuda,
    every served call cuda, no refused call."""
    try:
        import collections

        from foldjet.modules.engine.serving import report_observations

        rep = report_observations(predictor._network)["modules"]
        agg = collections.defaultdict(lambda: [collections.Counter(), collections.Counter(), collections.Counter(), 0])
        for m in rep.values():
            a = agg[m["class"]]
            a[0][str(m["backend"])] += 1
            a[1].update(m["served_host_calls"])
            a[2].update(m["refused_host_calls"])
            a[3] += 1
        all_cuda = True
        classes = {}
        for c, (s, v, r, k) in sorted(agg.items()):
            classes[c] = {"modules": k, "set": dict(s), "served": dict(v), "refused": dict(r)}
            all_cuda &= set(s) <= {"cuda"} and set(v) <= {"cuda"} and not r
        out = {"all_cuda": bool(all_cuda), "classes": classes}
        print("BACKEND_TABLE " + json.dumps(out, default=str), flush=True)
        return out
    except Exception as error:  # noqa: BLE001
        print("BACKEND_TABLE unavailable:", repr(error)[:200], flush=True)
        return None


def engine_source_version(engine_from: str | None) -> str | None:
    """Version of an engine source tree that shadows the installed engine (a release candidate): `version` of the pyproject.toml above `src/`."""
    if not engine_from:
        return None
    for parent in Path(engine_from).parents[:3]:
        toml = parent / "pyproject.toml"
        if toml.exists():
            for line in toml.read_text().splitlines():
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
    return None
