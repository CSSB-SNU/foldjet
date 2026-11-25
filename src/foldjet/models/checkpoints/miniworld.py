r"""The miniworld checkpoint format: an fp32 ``state_dict`` with engine-class keys.

The keys are the FoldJet graph's own (engine-class) keys. Every model's graph is built
from miniworld-engine classes, so its parameters have the engine's names and layouts.
Its checkpoint is that ``state_dict`` plus a format tag, saved with ``torch.save`` and
read with ``weights_only=True``::

    {"format": "miniworld", "version": 1, "family": "alphafold3",
     "source": "af3.bin.zst", "state_dict": {name: fp32 tensor}}

* **Loading** (:func:`load_into`) is ``model.load_state_dict(state_dict, strict=True)``:
  no translation tree.
* **Converting** (:func:`convert_haiku`, offline, once per released blob) is the only
  place the released layout is dealt with: the Haiku blob is read into the DENSE
  released-layout graph (:mod:`foldjet.models.checkpoints.legacy`,
  ``haiku.import_jax_weights_``), and every module the engine graph holds as an engine
  class is converted with the rules of
  :mod:`foldjet.models.checkpoints.legacy.released` (split / broadcast / discard).
  The conversion refuses to write a file unless every released key was either read,
  copied or a known discard (``released.DISCARDED``) and the result has exactly the
  keys and shapes of the engine graph.

The Fourier noise features (``diffusion_head.fourier_embeddings.{weight,bias}``, fixed
constants or trained, depending on the family) are buffers registered at load time; they
are in the file and set by :func:`load_into`.

    python -m foldjet.models.checkpoints.miniworld convert af3 \
        [--blob PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

FORMAT = "miniworld"
VERSION = 1
SUFFIX = ".miniworld.pt"

#: Registered model name -> the converted file in its checkpoint directory (the Haiku
#: blob it is converted from is ``DEFAULT_FILES``).
NATIVE_FILES = {
    "af3": "af3.miniworld.pt",
    "intellifold2": "intellifold2.miniworld.pt",
    "openfold3": "openbind0.miniworld.pt",
    "openfold3-preview2": "openfold3.miniworld.pt",
    "boltz2": "boltz2.miniworld.pt",
    "rosettafold3": "rosettafold3.miniworld.pt",
    "chai1": "chai1.miniworld.pt",
    "esmfold2": "esmfold2.miniworld.pt",
    "esmfold2-fast": "esmfold2_fast.miniworld.pt",
    "protenix1": "protenix1.miniworld.pt",
    "protenix2": "protenix2.miniworld.pt",
    "opendde": "opendde.miniworld.pt",
}

#: Buffers the graph registers at load time (not in a freshly built model's
#: ``state_dict``).
LOOSE = (
    "diffusion_head.fourier_embeddings.weight",
    "diffusion_head.fourier_embeddings.bias",
)


#: Released DiT-block tensors an engine block class may lack, droppable only when all
#: zero (see ``transformers._DROPPABLE_ZERO``).
DROPPABLE_ZERO = ("attention.to_query.bias",)

#: How many unread released keys a :class:`ConversionError` message lists.
_UNREAD_SHOWN = 8


class ConversionError(RuntimeError):
    """A released checkpoint that does not convert exactly.

    An unread key that is not a known discard, a missing / extra / mis-shaped key.
    """


def is_native(path: Path) -> bool:
    """Whether ``path`` names a miniworld-format file (by suffix)."""
    return path.name.endswith(SUFFIX)


# ------------------------------------------------------------------------------ loading
def read(path: Path) -> dict[str, Any]:
    """Return the checked contents of a miniworld-format file.

    Read with ``weights_only``; format tag and version verified.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        message = (
            f"{path} is not a miniworld-format checkpoint (format tag "
            f"{blob.get('format') if isinstance(blob, dict) else None!r})"
        )
        raise ValueError(message)
    if blob.get("version") != VERSION:
        message = (
            f"{path} is miniworld format version {blob.get('version')}; this code "
            f"reads version {VERSION}. Convert the release again."
        )
        raise ValueError(message)
    return blob


def load_into(model: nn.Module, path: Path, *, family: str) -> dict[str, Any]:
    """Strict load of a miniworld-format file into ``model``; returns a report.

    fp32 parameters; the Fourier buffers are registered.
    """
    blob = read(path)
    if blob["family"] != family:
        message = f"{path} holds family {blob['family']!r}, the model is {family!r}"
        raise ValueError(message)
    state = dict(blob["state_dict"])
    fourier = model.get_submodule("diffusion_head.fourier_embeddings")
    # registered first (their shapes are the file's), then loaded like every other entry
    for name in LOOSE:
        fourier.register_buffer(
            name.rsplit(".", 1)[1],
            torch.zeros_like(state[name], dtype=torch.float32).reshape(-1),
        )
        state[name] = state[name].to(torch.float32).reshape(-1)
    _adopt_dense_atom_stacks(model, state)
    model.load_state_dict(state, strict=True)
    return {
        "format": FORMAT,
        "format_version": VERSION,
        "source": blob.get("source"),
        "mapped_records": len(state),
    }


def _adopt_dense_atom_stacks(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Convert ESMFold2 atom stacks of an old file to the engine block layout, in place.

    A file written while ESMFold2's atom stacks were FoldJet's own (engine 2.5.1:
    ``cross_attention.i.*`` / ``transition_block.i.*``) loaded into a graph whose stacks
    are engine ``AtomDiT("esmfold2_pair")`` blocks (``blocks.i.*``): the same conversion
    as at convert time, in place. A file already in the block layout is left as it is.
    """
    from foldjet.models.checkpoints.legacy import (  # noqa: PLC0415 - deferred until called (import time)
        released,
    )
    from foldjet.modules.engine.transformers import (  # noqa: PLC0415 - deferred until called (import time)
        AtomDiTStack,
    )

    for name, module in model.named_modules():
        if (
            not (isinstance(module, AtomDiTStack) and module.rotary)
            or f"{name}.blocks.0.attention.to_query.weight" in state
        ):
            continue
        dense = {
            k[len(name) + 1 :]: v for k, v in state.items() if k.startswith(f"{name}.")
        }
        if not dense:
            continue
        for key in dense:
            del state[f"{name}.{key}"]
        # nn.Module attributes are dynamic (``__getattr__``): typed as Any
        first: Any = module.blocks[0]
        for i, tensors in enumerate(
            released.esmfold2_atom_blocks_from_released(
                dense, n_head=first.attention.n_head
            )
        ):
            for key, value in tensors.items():
                state[f"{name}.blocks.{i}.{key}"] = value


# --------------------------------------------------------------------------- converting
def _converters(engine_model: nn.Module) -> list[tuple[str, Any, nn.Module]]:  # noqa: C901 - one flat dispatch over the engine classes
    """``(prefix, function, module)`` of every engine-class module of ``engine_model``.

    ``function(tracked_state, prefix) -> {relative engine key: tensor}``; outermost
    modules only: parts of a converted module are converted with it.
    """
    from miniworld_engine.modules import (  # noqa: PLC0415 - deferred until called (import time)
        PairformerBlock,
        Transition,
    )

    from foldjet.models.checkpoints.legacy import (  # noqa: PLC0415 - deferred until called (import time)
        released,
    )
    from foldjet.modules.engine.graph import (  # noqa: PLC0415 - deferred until called (import time)
        EvoformerBlock,
    )
    from foldjet.modules.engine.transformers import (  # noqa: PLC0415 - deferred until called (import time)
        AtomDiTStack,
        TokenDiTStack,
    )

    def blocks(convert: Any) -> Any:
        def run(
            state: Mapping[str, torch.Tensor], prefix: str, module: nn.Module
        ) -> dict[str, torch.Tensor]:
            # nn.Module attributes are dynamic (``__getattr__``): the stack's blocks
            module_blocks = cast("nn.ModuleList", module.blocks)
            first: Any = module_blocks[0]
            if isinstance(module, TokenDiTStack):
                listed = convert(state, n_head=first.attention.n_head, prefix=prefix)
            # ESMFold2's atom block: no K / V AdaLN choice, no gate biases
            elif module.rotary:
                listed = released.esmfold2_atom_blocks_from_released(
                    state, n_head=first.attention.n_head, prefix=prefix
                )
            else:
                listed = convert(
                    state,
                    n_head=first.attention.n_head,
                    kv_norm=first.kv_norm,
                    prefix=prefix,
                )
            if len(listed) != len(module_blocks):
                message = (
                    f"{prefix}: the released state has {len(listed)} blocks, the "
                    f"graph {len(module_blocks)}"
                )
                raise ConversionError(message)
            result: dict[str, torch.Tensor] = {}
            for i, (block, tensors) in enumerate(
                zip(module_blocks, listed, strict=True)
            ):
                own = block.state_dict().keys()
                for k, v in tensors.items():
                    if k in own:
                        result[f"blocks.{i}.{k}"] = v
                    # RoseTTAFold3's bias-free query, released as an all-zero bias
                    elif k in DROPPABLE_ZERO and not bool(v.any()):
                        continue
                    else:
                        message = (
                            f"{prefix}.blocks.{i}: released tensor {k} has no "
                            f"parameter in {type(block).__name__}"
                            + (" and is not all zero" if k in DROPPABLE_ZERO else "")
                        )
                        raise ConversionError(message)
            return result

        return run

    token = blocks(released.token_dit_blocks_from_released)
    atom = blocks(released.atom_dit_blocks_from_released)
    found: list[tuple[str, Any, nn.Module]] = []
    taken: list[str] = []
    # every path, not every module object: a module shared under several indices
    # (RoseTTAFold3's one MSA block, ``spec.msa_shared_block``) is listed by
    # ``state_dict`` under each of them, so each is converted (and checked equal in
    # ``convert_state``); the de-duplicating default listed only ``msa_stack.0``
    for name, module in engine_model.named_modules(remove_duplicate=False):
        if not name or any(name.startswith(f"{p}.") for p in taken):
            continue
        if isinstance(module, PairformerBlock):
            fn = lambda s, p, _m: released.pairformer_block_from_released(s, prefix=p)
        elif isinstance(module, EvoformerBlock):
            fn = lambda s, p, _m: released.evoformer_block_from_released(s, prefix=p)
        elif isinstance(module, Transition) and re.fullmatch(
            r"diffusion_head\.(pair|single)_transition_\d", name
        ):
            fn = lambda s, p, _m: released.diffusion_transition_from_released(
                s, prefix=p
            )
        elif isinstance(module, TokenDiTStack):
            fn = token
        elif isinstance(module, AtomDiTStack):
            fn = atom
        else:
            continue
        taken.append(name)
        found.append((name, fn, module))
    return found


def _check_shared(
    first: tuple[str, dict[str, torch.Tensor]],
    prefix: str,
    converted: dict[str, torch.Tensor],
) -> None:
    """Refuse a shared module whose released tensors differ between its paths.

    One module under several paths (``first[0]`` converted first) holds one parameter
    set, so the released tensors of every path must convert to the same values.
    """
    if first[0] == prefix:
        return
    unequal = sorted(
        k
        for k in set(first[1]) | set(converted)
        if k not in first[1]
        or k not in converted
        or not torch.equal(first[1][k], converted[k])
    )
    if unequal:
        message = (
            f"{prefix} is the module of {first[0]} (shared), but the released "
            f"tensors differ: {unequal[:_UNREAD_SHOWN]}"
        )
        raise ConversionError(message)


def convert_state(
    released_state: Mapping[str, torch.Tensor], engine_model: nn.Module
) -> dict[str, torch.Tensor]:
    """Convert the DENSE released-layout ``state_dict`` to the engine graph's (fp32).

    Both graphs are of the same family. Raises :class:`ConversionError` unless: every
    released key under a converted module was read by its conversion or is a known
    discard (:func:`foldjet.models.checkpoints.legacy.released.is_discarded`), every
    other released key is a key of the engine graph, and the result has the engine
    graph's keys (the Fourier buffers aside) with their shapes.
    """
    from foldjet.models.checkpoints.legacy import (  # noqa: PLC0415 - deferred until called (import time)
        released,
    )

    tracked = released.Tracked(released_state)
    out: dict[str, torch.Tensor] = {}
    prefixes: list[str] = []
    shared: dict[int, tuple[str, dict[str, torch.Tensor]]] = {}
    for prefix, convert, module in _converters(engine_model):
        converted = convert(tracked, prefix, module)
        for key, value in converted.items():
            out[f"{prefix}.{key}"] = value
        prefixes.append(prefix)
        _check_shared(
            shared.setdefault(id(module), (prefix, converted)), prefix, converted
        )

    unread = sorted(
        k
        for k in released_state
        if any(k.startswith(f"{p}.") for p in prefixes)
        and k not in tracked.used
        and not released.is_discarded(k)
    )
    if unread:
        message = (
            "released keys neither read by a conversion nor a known discard: "
            f"{unread[:_UNREAD_SHOWN]}"
            f"{' ...' if len(unread) > _UNREAD_SHOWN else ''}"
        )
        raise ConversionError(message)
    target = engine_model.state_dict()
    for key, value in released_state.items():
        if any(key.startswith(f"{p}.") for p in prefixes):
            continue
        if key not in target and key not in LOOSE:
            message = (
                f"released key {key!r} is not a parameter of the engine graph and no "
                "conversion covers it"
            )
            raise ConversionError(message)
        out[key] = value

    missing = sorted(set(target) - set(out))
    extra = sorted(set(out) - set(target) - set(LOOSE))
    wrong = sorted(
        k
        for k in set(out) & set(target)
        if tuple(out[k].shape) != tuple(target[k].shape)
    )
    if missing or extra or wrong:
        message = (
            f"conversion does not match the engine graph: missing={missing[:6]}, "
            f"extra={extra[:6]}, mis-shaped={wrong[:6]}"
        )
        raise ConversionError(message)
    for key in LOOSE:
        if key not in out:
            message = f"the released graph has no {key}"
            raise ConversionError(message)
    return {
        k: v.detach().to(torch.float32).contiguous().clone() for k, v in out.items()
    }


def convert_haiku(
    blob: Path, family: str, out: Path, *, mode: str = "fast"
) -> dict[str, Any]:
    """Convert a released Haiku blob of ``family`` to a miniworld-format file.

    The file is written at ``out``; the blob has the ``af3.bin.zst`` layout; returns a
    report.
    """
    from foldjet.models.architectures.af3 import (  # noqa: PLC0415 - deferred until called (import time)
        AlphaFold3,
    )
    from foldjet.models.checkpoints.legacy import (  # noqa: PLC0415 - deferred until called (import time)
        dense_model,
    )
    from foldjet.models.checkpoints.legacy.haiku import (  # noqa: PLC0415 - deferred until called (import time)
        import_jax_weights_,
    )
    from foldjet.modules.dense.spec import (  # noqa: PLC0415 - deferred until called (import time)
        spec_for,
    )
    from foldjet.modules.engine.install import (  # noqa: PLC0415 - deferred until called (import time)
        skip_engine_init,
    )

    kwargs = {"num_recycles": 1, "num_samples": 1, "diffusion_steps": 1}
    spec = spec_for(family, mode)
    dense = dense_model(spec, **kwargs)
    mapped = import_jax_weights_(dense, blob)
    with skip_engine_init():
        engine = AlphaFold3(spec=spec, **kwargs)
    state = convert_state(dense.state_dict(), engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    torch.save(
        {
            "format": FORMAT,
            "version": VERSION,
            "family": family,
            "source": blob.name,
            "state_dict": state,
        },
        tmp,
    )
    tmp.replace(out)
    return {
        "out": str(out),
        "records": len(state),
        "source_records": mapped["source_records"],
        "format_version": VERSION,
    }


def convert_model(
    model: str, *, blob: Path | None = None, out: Path | None = None, mode: str = "fast"
) -> dict[str, Any]:
    """Convert registered ``model``'s released blob to its miniworld-format file.

    ``blob`` defaults to the model's checkpoint directory, ``out`` to the file next to
    the blob.
    """
    from foldjet.models import (  # noqa: PLC0415 - deferred until called (import time)
        entry,
    )
    from foldjet.models.checkpoints import (  # noqa: PLC0415 - deferred until called (import time)
        DEFAULT_FILES,
        resolve,
    )

    blob = blob or resolve(model, DEFAULT_FILES[model])
    out = out or blob.resolve().parent / NATIVE_FILES[model]
    return convert_haiku(blob, entry(model).family or "alphafold3", out, mode=mode)


def resolve_native(model: str) -> Path:
    """Return the miniworld-format file of registered ``model`` in its checkpoint dir.

    Never converts: a missing file raises with the exact command that makes it (the
    conversion is an offline, one-time step).
    """
    from foldjet.models.checkpoints import (  # noqa: PLC0415 - deferred until called (import time)
        DEFAULT_FILES,
        resolve,
    )

    if model not in NATIVE_FILES:
        message = (
            f"no miniworld-format checkpoint is defined for {model!r}; known: "
            f"{', '.join(sorted(NATIVE_FILES))}"
        )
        raise ValueError(message)
    try:
        return resolve(model, NATIVE_FILES[model])
    except FileNotFoundError as error:
        try:
            resolve(model, DEFAULT_FILES[model])
        except FileNotFoundError:
            step = (
                f"first fetch the released blob: foldjet weights download {model}; "
                f"then convert it: foldjet weights convert {model}"
            )
        else:
            step = (
                "the released blob is present; convert it once: foldjet weights "
                f"convert {model}"
            )
        message = (
            f"no miniworld-format checkpoint ({NATIVE_FILES[model]}) for {model!r}: "
            f"{step} (same as: python -m foldjet.models.checkpoints.miniworld "
            f"convert {model}). The loader never converts at run time.\n{error}"
        )
        raise FileNotFoundError(message) from error


def main(argv: list[str] | None = None) -> int:
    """``convert <model> [--blob PATH] [--out PATH]``: blob to miniworld-format file.

    The released blob of a registered model is converted to its miniworld-format file.
    """
    parser = argparse.ArgumentParser(
        prog="foldjet.models.checkpoints.miniworld",
        description=__doc__.splitlines()[0],  # pyright: ignore[reportOptionalMemberAccess]  # the module has a docstring
    )
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="Haiku blob -> miniworld format")
    convert.add_argument("model", choices=sorted(NATIVE_FILES))
    convert.add_argument(
        "--blob",
        type=Path,
        default=None,
        help="the released blob (default: the model's checkpoint directory)",
    )
    convert.add_argument(
        "--out",
        type=Path,
        default=None,
        help="the file to write (default: next to the blob)",
    )
    convert.add_argument("--mode", default="fast", choices=("fast", "exact"))
    arguments = parser.parse_args(argv)
    report = convert_model(
        arguments.model, blob=arguments.blob, out=arguments.out, mode=arguments.mode
    )
    print(report)  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
