"""Where FoldJet keeps what is not source: weights, databases, runs.

Nothing machine-specific lives in the repository. Everything below sits under
one home directory, ``$FOLDJET_HOME`` (default ``~/.cache/foldjet``), the way
the Hugging Face hub keeps its cache; each part can also be pointed elsewhere by
its own variable:

=================  ==========================  ================================
part               default                     override
=================  ==========================  ================================
converted weights  ``<home>/checkpoints/``     ``FOLDJET_CHECKPOINT_DIR``
CCD database       ``<home>/ccd/...lmdb``      ``FOLDJET_CCD_DB``
prediction runs    ``<home>/runs/``            ``FOLDJET_RUNS``
release envs       ``<home>/releases/``        ``FOLDJET_RELEASE_ROOT``
benchmark inputs   ``<home>/benchmarks/``      ``FOLDJET_BENCHMARKS``
=================  ==========================  ================================
"""

from __future__ import annotations

import os
from pathlib import Path

#: The repository checkout (``<repo>/src/foldjet/paths.py``); only for the
#: tracked files a checkout carries, such as the release references.
REPO = Path(__file__).resolve().parents[2]


def _env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def home() -> Path:
    """Return FoldJet's home directory.

    The pre-rename default ``~/.cache/foldforge`` keeps being used while no ``~/.cache/foldjet`` exists (the converted
    weights and databases there are not moved or rebuilt for the rename).
    """
    new, old = (
        Path("~/.cache/foldjet").expanduser(),
        Path("~/.cache/foldforge").expanduser(),
    )
    return _env("FOLDJET_HOME", old if old.is_dir() and not new.exists() else new)


def checkpoints() -> Path:
    """Return the converted-weights directory, one subdirectory per model."""
    return _env("FOLDJET_CHECKPOINT_DIR", home() / "checkpoints")


def ccd_db() -> Path:
    """Return the shared MiniWorld BioMol CCD database."""
    return _env("FOLDJET_CCD_DB", home() / "ccd" / "preprocessed_CCD.lmdb")


def runs() -> Path:
    """Return where a prediction goes when its --out is a bare run name."""
    return _env("FOLDJET_RUNS", home() / "runs")


def releases() -> Path:
    """Return the released implementations' environments."""
    return _env("FOLDJET_RELEASE_ROOT", home() / "releases")


def benchmarks() -> Path:
    """Return the benchmark input sets (replicated targets, databases)."""
    return _env("FOLDJET_BENCHMARKS", home() / "benchmarks")


#: The released implementations' reference structures, tracked with the tests.
RELEASE_REFERENCES = REPO / "tests" / "release_references"
