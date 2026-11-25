"""Import shim: MiniWorld's data modules import ``foldjet_engine.utils.typecheck`` (engine 2.4.x); the engine in the FoldJet env (2.5.x) no longer ships it.
In MiniWorld's engine that decorator is the identity unless ``settings.typecheck`` is on (default off), so an identity module is registered here.
Nothing is installed or written into any env.
"""

from __future__ import annotations

import sys
import types

if "foldjet_engine.utils.typecheck" not in sys.modules:
    try:
        import foldjet_engine.utils.typecheck  # noqa: F401  (a real one wins when present)
    except ImportError:
        module = types.ModuleType("foldjet_engine.utils.typecheck")
        module.typecheck = lambda obj: obj  # type: ignore[attr-defined]
        module.__all__ = ["typecheck"]
        sys.modules["foldjet_engine.utils.typecheck"] = module

# ``miniworld.training/__init__`` pulls in the whole trainer (param_policy -> miniworld.blocks -> an OLD ``foldjet.blocks`` that the current
# FoldJet does not have). The data pipeline only needs ``miniworld.training.batch`` (BaseBatch), so ``miniworld.training`` is registered as a
# bare package (same search path, its __init__ never runs).
if "miniworld.training" not in sys.modules:
    import importlib.util
    import os

    _spec = importlib.util.find_spec("miniworld")
    if _spec is not None and _spec.submodule_search_locations:
        _path = [os.path.join(p, "training") for p in _spec.submodule_search_locations]
        _pkg = types.ModuleType("miniworld.training")
        _pkg.__path__ = _path  # type: ignore[attr-defined]
        _pkg.__package__ = "miniworld.training"
        sys.modules["miniworld.training"] = _pkg
