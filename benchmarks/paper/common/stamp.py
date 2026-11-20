"""Write the run stamp of a paper experiment: FoldForge commit, engine version, GPU(s), node, Slurm job id, FP8 switches.

    python stamp.py OUT.json [key=value ...]      (run with the FoldForge env's python, on the compute node, inside the job)

Every result of benchmarks/paper carries this record (INSTRUCTIONS common rule: commit, engine version, GPU, node, job id).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f"unavailable: {error}"


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def stamp() -> dict:
    engine_file = None
    try:
        import miniworld_engine  # noqa: PLC0415

        engine_file = str(getattr(miniworld_engine, "__path__", [None])[0])
    except Exception:  # noqa: BLE001
        pass
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "foldforge_commit": _run("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "foldforge_dirty_paths": _run("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no").splitlines(),
        "foldforge_version": _version("foldforge"),
        "engine_version": _version("miniworld-engine"),
        "engine_kernels": {d: _version(d) for d in ("miniworld-engine-kernels-sm90-cu129", "miniworld-engine-kernels-sm80-cu129", "miniworld-engine-kernels-sm120-cu129")},
        "engine_path": engine_file,
        "gpus": _run("nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader").splitlines(),
        "node": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("MINIWORLD_", "FF_", "FOLDFORGE_")) or k == "ENGINE_OVERLAY"},
    }


if __name__ == "__main__":
    out = Path(sys.argv[1])
    rec = stamp()
    rec.update(dict(a.split("=", 1) for a in sys.argv[2:]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"stamp -> {out}: commit {rec['foldforge_commit'][:9]} engine {rec['engine_version']} node {rec['node']} job {rec['slurm_job_id']}")
