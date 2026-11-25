"""FoldJet's default for the CUDA caching allocator: expandable segments.

``import foldjet`` sets ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` when
neither ``PYTORCH_CUDA_ALLOC_CONF`` nor ``PYTORCH_ALLOC_CONF`` (torch's newer name for
the same setting) is in the environment. A variable that is present is never
touched, whatever its value, so ``PYTORCH_CUDA_ALLOC_CONF=`` (empty) is the way to
run with torch's own allocator defaults.

torch reads the variable when the CUDA caching allocator initialises (the first CUDA
allocation), not at ``import torch``, so setting it at ``import foldjet`` works
even after ``import torch``. When CUDA is already initialised the same setting is
applied to the live allocator too (``torch._C._accelerator_setAllocatorSettings``):
the segments allocated from then on are expandable.

Measured on H100 (AF3, engine 2.5.12, 2026-11-17, 3 targets of ~1,160 tokens with
CUDA graphs): the same speed or 1-2 % faster and 24 % less peak reserved memory.
"""

from __future__ import annotations

import os
import sys

#: The value FoldJet sets when the user set none.
DEFAULT_ALLOC_CONF = "expandable_segments:True"
_VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def apply_default_alloc_conf() -> bool:
    """Set the default allocator configuration unless the user set one.

    Returns True when FoldJet set it, False when a user's variable was kept.
    """
    if any(name in os.environ for name in _VARIABLES):
        return False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOC_CONF
    torch = sys.modules.get("torch")  # only when the caller imported it already
    if torch is not None and torch.cuda.is_initialized():
        # torch.cuda._set_allocator_settings is deprecated in favour of this one
        torch._C._accelerator_setAllocatorSettings(DEFAULT_ALLOC_CONF)  # noqa: SLF001
    return True


apply_default_alloc_conf()  # at `import foldjet`: the package imports this first
