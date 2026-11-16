"""Which parameters training does not update, and why: ONE table, applied by :func:`prepare.enable_training`.

FoldForge keeps every checkpoint key (the format does not change), but some parameters have no business in the optimizer:

* they have no path to the loss (the graph discards their output), so they never get a gradient; or
* the original model does not have them (the converter zero-fills the slot): training would grow capacity the original lacks; or
* the original keeps them fixed.

Each :class:`Rule` names a family set (``spec.family``; ``None`` = every family), a name pattern (``re.fullmatch`` on the parameter name),
a reason, and optionally ``zero_filled``: freeze the tensor only while it is exactly zero at load (a slot a later checkpoint fills with real
values is not frozen). The classification of the remaining gradient oddities of one step (data dependent, underflow) is in
:func:`classify_gradient`. Nothing here edits a weight: it sets ``requires_grad`` and the optimizer skips what does not require it.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from foldforge.modules.dense.spec import DenseSpec

from foldforge.train.config import LossWeights

_CONDITIONING = r"(?:evoformer_conditioning|diffusion_head\.atom_cross_att_encoder)"


@dataclass(frozen=True)
class Rule:
    """One freeze rule: parameters of ``families`` whose name fully matches ``pattern`` (a regex, or a function of the spec)."""

    key: str
    reason: str
    pattern: str | Callable[[DenseSpec], str | None]
    families: frozenset[str] | None = None
    #: freeze only a tensor that is exactly zero when training starts (a zero-filled slot of the original)
    zero_filled: bool = False
    #: the rule applies to this spec only when the predicate holds
    when: Callable[[DenseSpec], bool] | None = None
    #: name of a ``LossWeights`` field: the rule applies only while that loss weight is exactly 0 (the parameters have no other path to the loss, so a
    #: zero weight leaves them without a gradient: "zero-weight", not a defect). ``None`` = the rule does not depend on the loss weights.
    zero_weight: str | None = None

    def regex(self, spec: DenseSpec, weights: LossWeights | None = None) -> re.Pattern[str] | None:
        if self.families is not None and spec.family not in self.families:
            return None
        if self.when is not None and not self.when(spec):
            return None
        if self.zero_weight is not None and getattr(weights if weights is not None else LossWeights(), self.zero_weight) != 0.0:
            return None
        pattern = self.pattern(spec) if callable(self.pattern) else self.pattern
        return re.compile(pattern) if pattern else None


def _last_msa_block(spec: DenseSpec) -> str | None:
    """The last MSA block's MSA-side parameters: the block order is OPM first (AF3 / Protenix), so the MSA it updates is the next block's input
    and the last block's is thrown away; only its pair update reaches the loss."""
    if not spec.msa_layers or spec.msa_update_before_opm:
        return None
    return rf"evoformer\.msa_stack\.{spec.msa_layers - 1}\.(?:msa_attention1|msa_transition)\..*"


RULES: tuple[Rule, ...] = (
    Rule(
        "atom_pair_unused",
        "the atom-pair conditioning built by `_per_atom_conditioning` is discarded by `prepare` (AF3 discards it too); the `_1` twins carry the "
        "pair, and the converter copies the one original weight into both slots",
        rf"{_CONDITIONING}\.(?:single_to_pair_cond_(?:row|col)|embed_pair_(?:offsets|distances))\.weight",
    ),
    Rule(
        "last_msa_block",
        "the last MSA block's updated MSA is discarded (OPM-first order): these parameters have no path to the loss",
        _last_msa_block,
    ),
    Rule(
        "experimentally_resolved",
        "the MiniWorld recipe trains no experimentally-resolved loss (weight 0, `train/loss/confidence.py`): the head never gets a gradient "
        "(the `protenix` recipe gives it a weight and trains it)",
        r"confidence_head\.experimentally_resolved_(?:ln|logits)\..*",
        zero_weight="resolved_loss",
    ),
    Rule(
        "pae_zero_weight",
        "the PAE loss has weight 0 (Protenix `alpha_pae` 0.0 in the initial / fine-tuning-1 stages, configs_base.py:402): the PAE head's only path to the "
        "loss is that loss, so it gets no gradient. (Protenix itself keeps these parameters in Adam with an all-zero gradient and weight_decay 1e-8, "
        "i.e. a decay-only update; FoldForge leaves the pretrained PAE head untouched instead.)",
        r"confidence_head\.(?:pae_logits_ln|pae_logits|pae_inter_logits)\..*",
        zero_weight="pae_loss",
    ),
    Rule(
        "boltz2_contact_fourier",
        "Boltz-2 keeps the contact Fourier projection fixed (a frozen random feature map); FoldForge must not train it",
        r"(?:evoformer|confidence_head\.reembedding)\.contact_conditioning\.contact_fourier\..*",
        families=frozenset({"boltz2"}),
    ),
    # ESMFold2: slots the original does not have (no single track, no atom pair, no contact / bond re-embedding in the confidence head); the
    # converter zero-fills them. Frozen only while they are still exactly zero.
    Rule(
        "esmfold2_single_track",
        "ESMFold2 has no single track in the confidence pairformer and no single re-embedding: zero-filled slots, not in the original",
        r"confidence_head\.confidence_pairformer\.\d+\.single_(?:attention|transition)\..*"
        r"|evoformer\.(?:prev_single_embedding(?:_layer_norm)?\.(?:weight|bias)|single_activations\.weight)"
        r"|confidence_head\.reembedding\.(?:s_input_to_s\.weight|s_norm\.(?:weight|bias))",
        families=frozenset({"esmfold2", "esmfold2-fast"}),
        zero_filled=True,
    ),
    Rule(
        "esmfold2_atom_pair",
        "this FoldForge ESMFold2 atom stack has no atom pair (the released one has; engine R6): the pair slots are zero-filled, not in the original",
        rf"{_CONDITIONING}\.(?:pair_mlp_[123]|single_to_pair_cond_(?:row|col)_1|embed_pair_(?:offsets|distances)_1|embed_pair_offsets_valid"
        r"|embed_trunk_(?:pair|single)_cond)\.weight"
        r"|(?:diffusion_head\.atom_cross_att_(?:encoder|decoder)|evoformer_conditioning)\.atom_transformer_(?:encoder|decoder)\.pair_logits_projection\.weight",
        families=frozenset({"esmfold2", "esmfold2-fast"}),
        zero_filled=True,
    ),
    Rule(
        "esmfold2_confidence_contacts",
        "ESMFold2's confidence head has no contact / bond re-embedding: zero-filled slots, not in the original",
        r"confidence_head\.reembedding\.(?:contact_conditioning\..*|token_bonds_type_embed\.weight)",
        families=frozenset({"esmfold2", "esmfold2-fast"}),
        zero_filled=True,
    ),
)

#: Zero gradients that depend on the example (not a defect): the parameter needs charged atoms / covalent bonds in the example to receive one.
#: A smoke on a target with both (7fvu: 26 charged atoms, 41 ligand bonds) shows the gradient flowing.
DATA_DEPENDENT = re.compile(
    rf"{_CONDITIONING}\.embed_ref_charge\.weight|evoformer\.bond_embedding\.weight"
)

#: A zero gradient on a parameter this small, relative to the model's typical weight, is gradient underflow of a decayed (numerically dead)
#: module (released Protenix v1 weights decay to 1e-37 .. 1e-12), not a missing path; it is the same on the pytorch backend.
UNDERFLOW_RELATIVE = 1e-6


def frozen_parameters(model: nn.Module, weights: LossWeights | None = None) -> dict[str, Rule]:
    """``{parameter name: rule}`` of the parameters the policy freezes for ``model`` (its ``spec.family``) under the loss ``weights`` (``None`` = the
    MiniWorld recipe's defaults). Zero-filled rules read the weights of the model."""
    spec = model.spec
    rules = [(rule, regex) for rule in RULES if (regex := rule.regex(spec, weights)) is not None]
    found: dict[str, Rule] = {}
    for name, param in model.named_parameters():
        for rule, regex in rules:
            if regex.fullmatch(name) and not (
                rule.zero_filled and bool(param.detach().any())
            ):
                found[name] = rule
                break
    return found


def apply(model: nn.Module, weights: LossWeights | None = None) -> dict[str, int]:
    """Set ``requires_grad_(False)`` on the frozen parameters; returns ``{rule key: count}``."""
    counts: dict[str, int] = {}
    named = dict(model.named_parameters())
    for name, rule in frozen_parameters(model, weights).items():
        named[name].requires_grad_(False)
        counts[rule.key] = counts.get(rule.key, 0) + 1
    return counts


@torch.no_grad()
def weight_scale(model: nn.Module) -> float:
    """The median over the parameters of their largest absolute value: the model's typical weight size (the yardstick of ``classify_gradient``)."""
    sizes = [float(p.detach().abs().max()) for p in model.parameters() if p.numel()]
    return statistics.median(sizes) if sizes else 0.0


def classify_gradient(name: str, param: torch.Tensor, scale: float) -> str:
    """One trainable parameter after a step: ``ok`` | ``none`` (no gradient) | ``nonfinite`` | ``data_dependent`` | ``underflow`` | ``zero`` (CHECK)."""
    grad = param.grad
    if grad is None:
        return "none"
    if not bool(torch.isfinite(grad).all()):
        return "nonfinite"
    if bool(grad.any()):
        return "ok"
    if DATA_DEPENDENT.fullmatch(name):
        return "data_dependent"
    if float(param.detach().abs().max()) < UNDERFLOW_RELATIVE * scale:
        return "underflow"
    return "zero"
