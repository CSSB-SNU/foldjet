"""Environment-variable names: ``FOLDJET_*`` since the rename; the old ``FOLDFORGE_*`` names still work.

Imported first by ``foldjet/__init__.py``: each ``FOLDFORGE_X`` is copied to an unset ``FOLDJET_X`` (one ``FutureWarning``
names them). The engine maps its own ``MINIWORLD_*`` names the same way (``foldjet_engine._envcompat``).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import MutableMapping

OLD, NEW = "FOLDFORGE_", "FOLDJET_"


def sync(environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Copy old names to unset new names; return the old names that were copied."""
    copied = []
    for key in list(environ):
        if key.startswith(OLD) and NEW + key[len(OLD) :] not in environ:
            environ[NEW + key[len(OLD) :]] = environ[key]
            copied.append(key)
    return copied


_copied = sync()
if _copied:
    warnings.warn(
        "FoldJet renamed its environment variables FOLDFORGE_* -> FOLDJET_*; read as the new names: "
        + ", ".join(sorted(_copied)),
        FutureWarning,
        stacklevel=2,
    )
