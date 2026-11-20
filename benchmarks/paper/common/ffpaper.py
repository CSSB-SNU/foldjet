"""Shared helpers of the paper experiments F3 / F4 / F5 / F9 (inference timing, sample comparison, inference limit).

Import this module BEFORE ``foldforge`` / ``miniworld_engine`` in a runner: ``apply_arm_env`` sets the FP8 switches, which the engine reads when
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
CARDS = {"h100": "H100", "a100": "A100", "rtx5090": "5090"}

#: FoldForge predictors of F3 / F9 (the 5I28 benchmark's eleven) and of F4 / F5 (the FoldBench 27-target set's eleven; the two lists differ:
#: F4 / F5 have openfold3-preview2 and no esmfold2-fast).
PREDICTORS_5I28 = (
    "protenix1", "protenix2", "opendde", "esmfold2", "esmfold2-fast", "intellifold2", "af3", "boltz2", "chai1", "openfold3", "rosettafold3",
)
PREDICTORS_FOLDBENCH = (
    "af3", "boltz2", "chai1", "esmfold2", "intellifold2", "opendde", "openfold3", "openfold3-preview2", "protenix1", "protenix2", "rosettafold3",
)

_FP8_ENV_PREFIXES = ("MINIWORLD_",)


def apply_arm_env(arm: str, model: str) -> None:
    """Set (or clear) the FP8 environment for one arm. Call before importing the engine.

    ``engine_fp8``: ``MINIWORLD_FP8=safe`` + ``MINIWORLD_FP8_MODEL=<model>`` (the engine keeps TriAttn on the fp16 core for a model outside its
    ``TRIATTN_MODELS``, so the variable is set for every model) + lazy guards (no host sync inside a CUDA-graph capture; the saturation check
    ``fp8.check()`` runs once per cell by the runner). Every other arm: no FP8 variable at all.
    """
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    for key in [k for k in os.environ if k.startswith("MINIWORLD_") and "FP8" in k]:
        os.environ.pop(key)
    if arm == "engine_fp8":
        os.environ["MINIWORLD_FP8"] = "safe"
        os.environ["MINIWORLD_FP8_MODEL"] = model
        if os.environ.get("FFPAPER_FP8_GUARD", "lazy") == "lazy":
            for route in ("TRIMUL", "OPM", "PWA", "TRANSITION"):
                os.environ[f"MINIWORLD_{route}_FP8_GUARD"] = "lazy"


def card_of(gpu_name: str) -> str:
    low = gpu_name.lower()
    for key, needle in (("h100", "h100"), ("a100", "a100"), ("rtx5090", "5090")):
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
        import miniworld_engine

        engine_from = str(Path(miniworld_engine.__file__).parent)
    except Exception:  # noqa: BLE001
        engine_from = None
    try:
        engine = metadata.version("miniworld-engine")
    except metadata.PackageNotFoundError:
        engine = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    meta = {
        "foldforge_commit": _git(repo, "rev-parse", "HEAD"),
        "foldforge_dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=no")),
        "engine_version_installed": engine,
        "engine_loaded_from": engine_from,  # differs from the installed one when PYTHONPATH shadows it (a release candidate)
        "engine_source_override": os.environ.get("PYTHONPATH", "") or None,
        "torch": torch.__version__,
        "gpu": gpu,
        "card": card_of(gpu) if gpu else None,
        "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if gpu else None,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fp8_env": {k: v for k, v in os.environ.items() if k.startswith("MINIWORLD_") and "FP8" in k},
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
    if os.environ.get("MINIWORLD_FP8") is None:
        return None
    import miniworld_engine.fp8 as fp8

    return bool(fp8.check())


def fp8_status() -> dict | None:
    if os.environ.get("MINIWORLD_FP8") is None:
        return None
    import miniworld_engine.fp8 as fp8

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
