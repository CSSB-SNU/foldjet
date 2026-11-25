"""One eager training micro step of a FoldJet AlphaFold3-family model.

Protenix v1: single-stage (full) finetuning -- every parameter trains, the
confidence head jointly with the rest.

Order (MiniWorld recipe): trunk ``passes`` times -- every pass but the last under
``no_grad`` (the previous recycles are detached) -- then ``num_augment`` EDM noise
draws of the ground truth (random rotation / translation, one sigma per sample),
the diffusion head on the noisy coordinates, the diffusion (+ bond) loss, the
distogram loss on the trunk pair, and the backward of that sum (diffusion head +
distogram head + trunk). The confidence head then reads a structure from a
no-grad mini-rollout (``rollout_steps`` steps, one sample) with the trunk single /
pair / target features DETACHED (Protenix ``ConfidenceHead(stop_gradient=True)``,
confidence.py:178-181; MiniWorld ``benchmarks/protenix_finetune/step.py:219``):
the confidence loss trains the confidence head only, the trunk and the diffusion
head learn from the diffusion and distogram losses, so the two backwards are
independent and their gradients simply add up in ``param.grad``.

``cfg.rollout`` places the mini-rollout: ``after`` the trunk backward (default),
``before`` the diffusion forward, or on a second CUDA stream (``side``, MiniWorld
``--side_conf``) overlapped with the diffusion-train forward / backward; the
confidence head runs after the join. Activations run in bf16 under
``activation_autocast`` (fp32 master parameters); the losses are fp32.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch

from foldjet.modules.engine.configure import activation_autocast, diffusion_autocast
from foldjet.train.loss import atom_weights, bond_loss, diffusion_mse, distogram_loss
from foldjet.train.loss.confidence import confidence_loss, confidence_targets
from foldjet.train.loss.diffusion import random_rotation_translation, sample_sigma
from foldjet.train.profile import PhaseTimer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from foldjet.train.config import TrainConfig
    from foldjet.train.data import TrainBatch


@contextmanager
def foreach_accumulation(params: list[torch.nn.Parameter]) -> Iterator[None]:
    """Run a backward with the parameters' ``.grad`` detached, then add.

    The new gradients go into the old ones with ONE ``torch._foreach_add_``.

    Autograd's ``AccumulateGrad`` adds a new gradient into an existing ``.grad``
    with one ``add_`` launch per parameter (``grad += new``: 4.4k launches, ~8.5 ms
    of a Protenix v1 micro step in a step graph). Here every ``.grad`` is set to
    ``None`` for the backward, so autograd stores the new gradient itself (no
    kernel; exactly what it does for a parameter without a gradient), and the old
    buffers get ``old + new`` from the multi-tensor add: the same fp32 addition per
    element with the same operands (``a + 1 * b``), so the result is bitwise the
    same. A parameter without an old gradient keeps the new one, as before. The
    ``.grad`` tensor objects are the old ones afterwards (a step graph accumulates
    into them). ``FOLDFORGE_FOREACH_GRAD_ACCUM=0``: plain autograd accumulation.
    """
    if os.environ.get("FOLDFORGE_FOREACH_GRAD_ACCUM", "1") == "0":
        yield
        return
    held = [p.grad for p in params]
    for p in params:
        p.grad = None
    done = False
    try:
        yield
        done = True
    finally:
        dst, src = [], []
        for p, old in zip(params, held, strict=True):
            # The first gradient of this parameter: autograd stored it as it does
            # without this context.
            if old is None:
                continue
            new = p.grad
            p.grad = old
            if done and new is not None:
                dst.append(old)
                src.append(new)
        if dst:
            torch._foreach_add_(dst, src)  # pyright: ignore[reportPrivateImportUsage]  # public torch op


def trainable(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return the parameters of ``model`` that require a gradient, in order."""
    return [p for p in model.parameters() if p.requires_grad]


class DeviceParts:
    """Loss parts of one micro step as DEVICE scalars.

    No host read inside the step: every ``float(tensor)`` is a host sync that
    drains the CUDA queue and blocks graph capture. :meth:`floats` reads them ONCE
    and replays the host accumulation of the old per-chunk ``float()`` code in the
    same order (double arithmetic on the exact fp32 values), so the numbers are
    bit-identical to it.
    """

    def __init__(self, passes: int, share_weights: tuple[float, float, float]) -> None:
        self.passes = passes
        self.w_diffusion, self.w_bond, self.w_distogram = share_weights
        # Per chunk: share, l_diffusion, l_bond, chunk_loss.
        self.chunks: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.distogram: torch.Tensor | None = None
        self.confidence: torch.Tensor | None = None
        self.conf_parts: dict[str, torch.Tensor] = {}
        # ``only`` modes: "loss" (trunk dummy loss) etc., read as is
        self.plain: dict[str, torch.Tensor] = {}

    def tensors(self) -> list[torch.Tensor]:
        """Return the device scalars in the order :meth:`floats` reads them."""
        flat: list[torch.Tensor] = []
        for _, d, b, c in self.chunks:
            flat += [d, b, c]
        for t in (self.distogram, self.confidence):
            if t is not None:
                flat.append(t)
        flat += list(self.conf_parts.values())
        flat += list(self.plain.values())
        return flat

    def floats(self) -> dict[str, float]:
        """Read the parts with ONE host read and accumulate them as host floats."""
        flat = self.tensors()
        # the ONE host read
        values = (
            torch.stack([t.detach().reshape(()).float() for t in flat]).tolist()
            if flat
            else []
        )
        it = iter(values)
        out: dict[str, float] = {
            "loss": 0.0,
            "diffusion": 0.0,
            "bond": 0.0,
            "distogram": 0.0,
        }
        for share, _, _, _ in self.chunks:
            d, b, c = next(it), next(it), next(it)
            out["diffusion"] += d * share
            out["bond"] += b * share
            out["loss"] += c
        if self.distogram is not None:
            out["distogram"] = next(it)
            out["loss"] += self.w_distogram * out["distogram"]
        if self.confidence is not None:
            conf = next(it)
            out["loss"] += conf
            out["confidence"] = conf
            out.update({k: next(it) for k in self.conf_parts})
        for k in self.plain:
            out[k] = next(it)
        out["passes"] = self.passes
        return out


def run_trunk_train(
    model: Any, batch_data: Any, passes: int, timer: PhaseTimer | None = None
) -> dict[str, torch.Tensor]:
    """Run ``AlphaFold3._run_trunk`` with gradient on the last pass only.

    Earlier recycles are detached; no MC dropout.
    """
    timer = timer or PhaseTimer(False)  # noqa: FBT003 - PhaseTimer's enabled flag
    target_feat, structure_feat = model.create_target_feat_embedding(batch_data)
    embeddings = {
        "pair": model._ssm_state(  # noqa: SLF001 - the model's initial pair state
            batch_data.num_res, target_feat.device
        ),
        "single": torch.zeros(
            [batch_data.num_res, model.evoformer_seq_channel],
            dtype=torch.float32,
            device=target_feat.device,
        ),
        "target_feat": target_feat,
        "structure_target_feat": structure_feat,
    }
    presample = (
        getattr(model.evoformer, "foldjet_msa_presampled", False)
        and model.evoformer.msa_stack_num_layer
    )
    for index in range(passes):
        last = index == passes - 1
        pass_batch = (
            replace(batch_data, msa=model.evoformer.sample_msa(batch_data.msa))
            if presample
            else batch_data
        )
        with (
            timer.phase(f"trunk_fwd_r{index}"),
            nullcontext() if last else torch.no_grad(),
        ):
            embeddings = model.evoformer(
                batch=pass_batch,
                prev=embeddings,
                target_feat=target_feat,
                first_pass=index == 0,
                last_pass=last,
                recycle_dropout=0.0,
            )
            if model.reference_precision:
                embeddings["pair"] = embeddings["pair"].float()
                embeddings["single"] = embeddings["single"].float()
    return embeddings


def mini_rollout(
    model: Any,
    features: Any,
    batch_data: Any,
    detached: dict[str, torch.Tensor],
    steps: int,
    graph: bool = False,  # noqa: FBT001, FBT002 - public signature
) -> torch.Tensor:
    """Sample a no-grad structure ``[res, atoms, 3]`` of the diffusion head.

    The (current) diffusion head runs from the detached trunk outputs. ``graph``:
    one CUDA graph per shape.
    """
    from foldjet.train import step_graph  # noqa: PLC0415 - step_graph imports step

    # Inside a step graph's warm-up / capture: the rollout as plain ops (no host
    # read, no graph of its own).
    if step_graph.inline_rollout(model):
        from foldjet.train import rollout_graph  # noqa: PLC0415 - lazy graph module

        return rollout_graph.rollout_body(
            model, rollout_graph.rollout_batch(batch_data), detached, steps
        )
    if graph:
        from foldjet.train import rollout_graph  # noqa: PLC0415 - lazy graph module

        return rollout_graph.mini_rollout(
            model, features, batch_data, detached, steps, mini_rollout
        )
    head = model.diffusion_head
    saved = (model.num_samples, model.diffusion_steps)
    head.eval()
    try:
        with torch.no_grad():
            rollout = model._prepare_diffusion_inputs(  # noqa: SLF001 - model internals
                features, batch_data, detached
            )
            model.num_samples, model.diffusion_steps = 1, steps
            return model._sample_diffusion(  # noqa: SLF001 - model internals
                rollout.batch, rollout.embeddings
            )["atom_positions"][0]
    finally:
        model.num_samples, model.diffusion_steps = saved
        head._fold = None  # noqa: SLF001 - the rollout's fold-scoped conditioning
        head.train()


def confidence_step(
    model: Any,
    features: dict[str, torch.Tensor],  # noqa: ARG001 - kept for the call signature
    batch_data: Any,
    detached: dict[str, torch.Tensor],
    x_mini: torch.Tensor,
    x0: torch.Tensor,
    atom_mask: torch.Tensor,
    cfg: TrainConfig,
    example: TrainBatch | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the confidence head on the rollout structure ``x_mini``.

    Trunk inputs are detached. Returns the pLDDT / PDE / PAE cross entropies
    against the targets of ``x_mini`` and the ground truth ``x0``
    ``[res, atoms, 3]``.
    """
    tf = batch_data.token_features
    head_batch = model._head_batch(batch_data)  # noqa: SLF001 - model internals
    gather = head_batch.pseudo_beta_info.token_atoms_to_pseudo_beta
    outputs = model.confidence_head(
        dense_atom_positions=x_mini.detach(),
        embeddings=detached,
        seq_mask=tf.mask,
        token_atoms_to_pseudo_beta=gather,
        asym_id=tf.asym_id,
        batch=head_batch,
    )
    targets = confidence_targets(
        x_mini,
        x0,
        atom_mask,
        gather,
        tf.mask,
        tf.aatype,
        tf.is_dna.bool() | tf.is_rna.bool(),
        tf.is_ligand,
        model.confidence_head,
        present_index=None if example is None else example.present_index,
        present_valid=None if example is None else example.present_valid,
        real_mask=batch_data.predicted_structure_info.atom_mask.bool()
        if cfg.loss.resolved_loss != 0.0
        else None,
        with_pae=cfg.loss.pae_loss != 0.0,
        protenix_targets=cfg.targets_protenix,
    )
    w = cfg.loss
    return confidence_loss(
        outputs,
        # a None target belongs to a zero-weight term, which confidence_loss never reads
        targets,  # pyright: ignore[reportArgumentType]
        plddt_loss=w.plddt_loss,
        pde_loss=w.pde_loss,
        pae_loss=w.pae_loss,
        resolved_loss=w.resolved_loss,
    )


def micro_step(
    model: Any,
    example: TrainBatch,
    cfg: TrainConfig,
    passes: int,
    *,
    scale: float = 1.0,
    timer: PhaseTimer | None = None,
) -> dict[str, float]:
    """Run :func:`micro_step_device` + the ONE host read of the loss parts.

    The host read happens at its end (the callers want floats).
    """
    return micro_step_device(
        model, example, cfg, passes, scale=scale, timer=timer
    ).floats()


def micro_step_device(  # noqa: C901, PLR0912, PLR0915 - one ordered step, kept linear
    model: Any,
    example: TrainBatch,
    cfg: TrainConfig,
    passes: int,
    *,
    scale: float = 1.0,
    timer: PhaseTimer | None = None,
) -> DeviceParts:
    """Forward + loss + backward of one example; returns the loss parts.

    Gradients accumulate into ``param.grad`` (scaled by ``scale``).

    The heads read LEAF copies of the trunk outputs, so the backward runs in
    independent pieces whose gradients add up: the diffusion head
    (``augment_chunk`` samples at a time -- a smaller chunk holds fewer diffusion
    activations), the distogram head, then ONE backward through the trunk with the
    accumulated gradient of the leaves, then the confidence path. ``cfg.rollout``
    places the mini-rollout: ``before`` (right after the trunk forward: its
    transient memory stacks on the saved trunk activations), ``side`` (on a second
    CUDA stream, overlapped with the diffusion-train forward / backward) or
    ``after`` (after the trunk backward, when the activations are freed).
    """
    # The whole micro step as a CUDA graph (train/step_graph.py); None = not served
    # (CPU, --only, a failed capture): run it eagerly.
    if cfg.step_graph:
        from foldjet.train import step_graph  # noqa: PLC0415 - import cycle

        if step_graph.mode(model) is None:
            served = step_graph.micro_step_device(
                model, example, cfg, passes, scale=scale
            )
            if served is not None:
                return served
    timer = timer or PhaseTimer(False)  # noqa: FBT003 - PhaseTimer's enabled flag
    device = example.atom_positions.device
    sigma_data = cfg.noise.sigma_data
    n = cfg.num_augment
    chunk = cfg.augment_chunk if 0 < cfg.augment_chunk < n else n
    # "" = the whole step; "trunk" | "diffusion" | "rollout" | "confidence" = one
    # part alone (memory / time measurements)
    only = cfg.only
    mode = cfg.rollout if cfg.confidence and not only else "none"
    # "graph": the rollout graph (train/rollout_graph.py) runs at the "after"
    # placement
    mode = "after" if mode == "graph" else mode
    side_stream = (
        torch.cuda.Stream(device) if mode == "side" and device.type == "cuda" else None
    )
    if mode == "side" and side_stream is None:
        mode = "after"
    box: dict[str, torch.Tensor] = {}
    out = DeviceParts(
        passes, (cfg.loss.diffusion_loss, cfg.loss.bond_loss, cfg.loss.distogram_loss)
    )
    # The backwards accumulate into their .grad with one multi-tensor add each
    # (foreach_accumulation).
    params = trainable(model)

    with activation_autocast(model):
        features = {
            k: (v.float() if v.dtype == torch.float64 else v)
            for k, v in example.features.items()
        }
        batch_data = model._prepare_batch(features)  # noqa: SLF001 - model internals
        with (
            torch.no_grad()
            if only in ("diffusion", "rollout", "confidence")
            else nullcontext()
        ):
            embeddings = run_trunk_train(model, batch_data, passes, timer)
        detached = {
            k: (v.detach() if torch.is_tensor(v) else v) for k, v in embeddings.items()
        }
        leaves = {
            k: v.detach().requires_grad_(True)  # noqa: FBT003 - torch API flag
            for k, v in embeddings.items()
            if torch.is_tensor(v) and v.requires_grad
        }
        head_inputs = {**embeddings, **leaves}
        # Trunk forward + backward alone: a dummy loss on its outputs stands for the
        # heads' gradient.
        if only == "trunk":
            dummy = cast(
                "torch.Tensor",
                sum(
                    v.float().pow(2).mean()
                    for v in embeddings.values()
                    if torch.is_tensor(v) and v.requires_grad
                ),
            )
            with timer.phase("trunk_bwd"), foreach_accumulation(params):
                (dummy * 1e-6 * scale).backward()
            out.plain["loss"] = dummy.detach()
            return out
        if only in ("rollout", "confidence"):
            with timer.phase("mini_rollout"):
                box["x_mini"] = mini_rollout(
                    model,
                    features,
                    batch_data,
                    detached,
                    cfg.rollout_steps,
                    graph=cfg.rollout == "graph",
                )
            if only == "rollout":
                return out
            x0 = example.atom_positions.float()
            amask = (
                example.atom_mask
                if example.atom_mask is not None
                else batch_data.predicted_structure_info.atom_mask
            ).bool()
            with timer.phase("confidence_fwd"):
                l_conf, parts = confidence_step(
                    model,
                    features,
                    batch_data,
                    detached,
                    box["x_mini"],
                    x0,
                    amask,
                    cfg,
                    example,
                )
            with timer.phase("confidence_bwd"), foreach_accumulation(params):
                (l_conf * scale).backward()
            out.confidence = l_conf.detach()
            out.conf_parts = {k: v.detach() for k, v in parts.items()}
            return out
        diffusion = model._prepare_diffusion_inputs(  # noqa: SLF001 - model internals
            features, batch_data, head_inputs
        )
        if mode == "before":
            with timer.phase("mini_rollout"):
                box["x_mini"] = mini_rollout(
                    model, features, batch_data, detached, cfg.rollout_steps
                )
        elif mode == "side":
            # mode "side" implies a side stream (it falls back to "after" without one)
            side_stream.wait_stream(torch.cuda.current_stream(device))  # pyright: ignore[reportOptionalMemberAccess]
            with torch.cuda.stream(side_stream), timer.phase("mini_rollout"):
                box["x_mini"] = mini_rollout(
                    model, features, batch_data, detached, cfg.rollout_steps
                )

        head = model.diffusion_head
        atom_mask = (
            example.atom_mask
            if example.atom_mask is not None
            else batch_data.predicted_structure_info.atom_mask
        ).bool()
        x0 = example.atom_positions.float()
        tf = batch_data.token_features
        weights = atom_weights(
            tf.is_dna,
            tf.is_rna,
            tf.is_ligand,
            atom_mask,
            alpha_dna=cfg.loss.alpha_dna,
            alpha_rna=cfg.loss.alpha_rna,
            alpha_ligand=cfg.loss.alpha_ligand,
        )
        w = cfg.loss
        # diffusion head (+ bond): ``chunk`` augmentations at a time, one backward per
        # chunk (the leaves collect the trunk-side gradient)
        for first in range(0, n, chunk):
            m = min(chunk, n - first)
            share = m / n
            with timer.phase("diffusion_train_fwd"):
                sigma = sample_sigma(cfg.noise, m, device)
                x0_aug = random_rotation_translation(
                    x0[None].expand(m, *x0.shape).contiguous(),
                    cfg.noise.translation_noise,
                )
                noise = torch.randn_like(x0_aug) * sigma.view(m, 1, 1, 1)
                x_noisy = (x0_aug + noise) * atom_mask[None, :, :, None]
                # the head runs under the model's diffusion autocast: fp32 for the
                # af3_default precision (AF3 / Protenix train the diffusion module in
                # fp32), the activation autocast otherwise (as inference and rollout)
                with diffusion_autocast(model):
                    table = head.conditioning_table(
                        diffusion.batch, diffusion.embeddings, sigma
                    )
                    x_pred = head(
                        x_noisy,
                        sigma,
                        diffusion.batch,
                        diffusion.embeddings,
                        True,  # noqa: FBT003 - the head's positional training flag
                        single_cond=table,
                    )
                # the fold's conditioning holds the autograd graph of this chunk
                head._fold = None  # noqa: SLF001 - drop the fold-scoped conditioning
                l_diffusion = diffusion_mse(
                    x_pred, x0_aug, sigma, atom_mask, weights, sigma_data
                )
                l_bond = bond_loss(
                    x_pred, x0_aug, sigma, example.bond_pairs, sigma_data
                )
                chunk_loss = (
                    w.diffusion_loss * l_diffusion + w.bond_loss * l_bond
                ) * share
            with timer.phase("diffusion_train_bwd"), foreach_accumulation(params):
                (chunk_loss * scale).backward()
            out.chunks.append(
                (share, l_diffusion.detach(), l_bond.detach(), chunk_loss.detach())
            )
            del x_pred, x_noisy, x0_aug, noise, table, chunk_loss, l_diffusion, l_bond
        with timer.phase("distogram_fwd_bwd"):
            logits = model.distogram_head(
                model._head_batch(batch_data),  # noqa: SLF001 - model internals
                head_inputs,
            )["logits"]
            l_disto = distogram_loss(
                logits,
                model.distogram_head.breaks,
                x0,
                atom_mask,
                batch_data.pseudo_beta_info.token_atoms_to_pseudo_beta,
                tf.mask,
            )
            with foreach_accumulation(params):
                (w.distogram_loss * l_disto * scale).backward()
        out.distogram = l_disto.detach()

    # ONE backward through the trunk with the gradient the heads left on the leaves
    with timer.phase("trunk_bwd"):
        # A leaf no head sent a gradient to (``grad is None``) contributes nothing to
        # the trunk backward. ``target_feat`` is expected to be one: the diffusion
        # head reads ``structure_target_feat`` where the family has one
        # (diffusion_head.py: ``embeddings.get("structure_target_feat",
        # embeddings["target_feat"])``; Protenix v1 does), so no head consumes
        # ``target_feat``. Any other missing leaf is named in the log.
        expected = {"target_feat"} if "structure_target_feat" in embeddings else set()
        missing = sorted(
            k for k in leaves if leaves[k].grad is None and k not in expected
        )
        if missing:
            logging.getLogger("foldjet.train").warning(
                "no gradient reached the trunk output(s): %s", ", ".join(missing)
            )
        used = [k for k in leaves if leaves[k].grad is not None]
        if used:
            with foreach_accumulation(params):
                torch.autograd.backward(
                    [embeddings[k] for k in used], [leaves[k].grad for k in used]
                )
    del leaves, head_inputs, diffusion, embeddings

    if cfg.confidence and not only:
        with activation_autocast(model):
            if mode == "side":
                torch.cuda.current_stream(device).wait_stream(side_stream)  # pyright: ignore[reportArgumentType]  # mode "side" implies a stream
                box["x_mini"].record_stream(torch.cuda.current_stream(device))
            elif mode == "after":
                with timer.phase("mini_rollout"):
                    box["x_mini"] = mini_rollout(
                        model,
                        features,
                        batch_data,
                        detached,
                        cfg.rollout_steps,
                        graph=cfg.rollout == "graph",
                    )
            with timer.phase("confidence_fwd"):
                l_conf, parts = confidence_step(
                    model,
                    features,
                    batch_data,
                    detached,
                    box["x_mini"],
                    x0,
                    atom_mask,
                    cfg,
                    example,
                )
        with timer.phase("confidence_bwd"), foreach_accumulation(params):
            (l_conf * scale).backward()
        out.confidence = l_conf.detach()
        out.conf_parts = {k: v.detach() for k, v in parts.items()}
    return out
