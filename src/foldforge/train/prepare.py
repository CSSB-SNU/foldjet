"""Switch a loaded (inference) FoldForge model to training mode.

``configure_model`` leaves the model in ``eval()`` with ``requires_grad_(False)``
and ``load_model`` marks dead (numerically zero) modules so inference can skip
them; both are undone here. Dead modules MUST run in training (their parameters
train).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch

from foldforge.modules.dense.dead import clear_dead
from foldforge.train import freeze

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from foldforge.train.config import TrainConfig


def enable_training(model: nn.Module, cfg: TrainConfig) -> dict[str, object]:
    """Switch to ``train()`` with gradients on for every non-frozen parameter.

    The frozen ones (``train/freeze.py``) are parameters with no path to the loss,
    slots the original model lacks and parameters the original keeps fixed. Dead
    skip goes off, and the distogram / confidence logits are kept for the loss.
    """
    cleared = clear_dead(model)
    model.train()
    model.requires_grad_(True)  # noqa: FBT003 - torch API positional flag
    frozen = freeze.apply(model, cfg.loss)
    # No per-shape inference sampler graph in training (static_weights / refresh
    # would go stale); rollout="graph" sets its own inline scope
    # (train/rollout_graph.py).
    model.foldforge_sampler_scope = None
    # The head returns its logits only when asked.
    model.confidence_head.save_logits = cfg.confidence
    # The head returns its logits only when asked.
    model.distogram_head.save_distogram = True
    return {
        "dead_cleared": cleared,
        "frozen": frozen,
        "sampler_ops_off_modules": disable_inference_only_ops(model),
    }


def draw_passes(cfg: TrainConfig, rng: random.Random) -> int:
    """Total trunk passes of one step, uniform in [recycles_min, recycles_max]."""
    return rng.randint(cfg.recycles_min, cfg.recycles_max)


def seed_everything(seed: int) -> random.Random:
    """Seed torch (CPU and every CUDA device); return a ``random.Random(seed)``."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return random.Random(seed)


def disable_inference_only_ops(model: nn.Module) -> int:
    """Turn off the sampler's inference-only fused rigid-motion glue.

    The glue (``AugmentNoise`` / ``AlignEuler``) has no autograd: set the
    module-level ``foldforge_sampler_glue = False`` (the switch ``af3`` reads). The
    atom encoder's ``PoolAtoms`` stays on: it is differentiable in ``x`` since
    engine 2.5.3 (R17), so training no longer needs the torch gather. Returns the
    module count.
    """
    count = 0
    for module in model.modules():
        module.foldforge_sampler_glue = False
        count += 1
    return count


#: Block classes whose forward is recomputed in the backward under activation
#: checkpointing (the pair-stack blocks that dominate the activation memory: the
#: engine ``PairformerBlock`` of the trunk / templates / confidence head and
#: FoldForge's ``EvoformerBlock``).
CHECKPOINTED_BLOCKS = ("PairformerBlock", "EvoformerBlock")

#: Block groups a policy can name (by the module path of the block's list:
#: ``trunk_pairformer.N``, ``msa_stack.N``, ``tmpl_pairformer`` /
#: ``template_embedding_iteration``, ``confidence_pairformer``).
CKPT_GROUPS = ("trunk", "msa", "template", "confidence")


def _block_group(name: str) -> str:
    if "confidence" in name:
        return "confidence"
    if "tmpl" in name or "template" in name:
        return "template"
    if "msa" in name:
        return "msa"
    return "trunk"


def ckpt_select(groups: dict[str, list[str]], policy: str) -> set[str]:
    """Return the block names ``policy`` wraps.

    ``groups`` maps group -> block names in module order. Policies: ``all`` (every
    block, the default), ``none``, ``pairformer`` (trunk blocks only),
    ``evoformer`` (MSA blocks only), ``except:<group>`` (all but one group:
    trunk / msa / template / confidence), and ``every:N`` / ``first:K`` /
    ``last:K``: those trunk pairformer blocks (index % N == 0 / the first K / the
    last K) PLUS every block of the other groups.
    """
    names = {n for blocks in groups.values() for n in blocks}
    trunk = groups.get("trunk", [])
    others = names - set(trunk)
    kind, _, arg = policy.partition(":")
    if kind == "all" and not arg:
        return names
    if kind == "none" and not arg:
        return set()
    if kind == "pairformer" and not arg:
        return set(trunk)
    if kind == "evoformer" and not arg:
        return set(groups.get("msa", []))
    if kind == "except" and arg in CKPT_GROUPS:
        return names - set(groups.get(arg, []))
    if kind in ("every", "first", "last") and arg.isdigit() and int(arg) > 0:
        k = int(arg)
        if kind == "every":
            chosen = trunk[::k]
        elif kind == "first":
            chosen = trunk[:k]
        else:
            chosen = trunk[len(trunk) - k :] if k < len(trunk) else trunk
        return others | set(chosen)
    msg = (
        f"unknown checkpoint policy {policy!r} (all | none | pairformer | "
        f"evoformer | except:<{'|'.join(CKPT_GROUPS)}> | every:N | first:K | "
        "last:K)"
    )
    raise ValueError(msg)


def enable_activation_checkpointing(model: nn.Module, policy: str = "all") -> int:
    """Wrap the forward of the selected pair-stack blocks in a checkpoint.

    ``policy`` (default ``all``: every block) selects the blocks; each forward runs
    under ``torch.utils.checkpoint`` (non-reentrant, RNG preserved) so its
    activations are recomputed in the backward instead of stored. Returns the
    number of blocks wrapped. Training only: ``torch.is_grad_enabled()`` is the
    guard.
    """
    from torch.utils.checkpoint import checkpoint  # noqa: PLC0415 - lazy, training only

    blocks: dict[str, nn.Module] = {}
    groups: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if type(module).__name__ not in CHECKPOINTED_BLOCKS or getattr(
            module, "_ff_checkpointed", False
        ):
            continue
        blocks[name] = module
        groups.setdefault(_block_group(name), []).append(name)
    chosen = ckpt_select(groups, policy)
    count = 0
    for name, module in blocks.items():
        if name not in chosen:
            continue
        forward = module.forward

        def wrapped(
            *args: Any, _forward: Callable[..., Any] = forward, **kwargs: Any
        ) -> Any:
            if torch.is_grad_enabled():
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)
            return _forward(*args, **kwargs)

        module.forward = wrapped
        module._ff_checkpointed = True  # noqa: SLF001 - marker on the wrapped block
        count += 1
    return count
