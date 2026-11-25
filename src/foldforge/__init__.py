"""Compatibility alias: ``FoldForge`` was renamed ``FoldJet``.

``import foldforge`` and ``import foldforge.<sub>`` return the SAME module objects as ``foldjet`` /
``foldjet.<sub>`` (one module per name, no second copy of any state), with one ``FutureWarning`` per process. New code
imports ``foldjet``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import foldjet as _new

_OLD, _NEW = "foldforge", "foldjet"


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``foldforge.x.y`` -> the module ``foldjet.x.y`` (imported once, shared under both names)."""

    def find_spec(self, name, path=None, target=None):  # noqa: ARG002
        if name.startswith(_OLD + "."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module(_NEW + spec.name[len(_OLD) :])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
warnings.warn(
    "foldforge was renamed foldjet; import foldjet instead",
    FutureWarning,
    stacklevel=2,
)
sys.modules[_OLD] = _new
