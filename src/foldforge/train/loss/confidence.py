"""Confidence-head loss of the single-stage (full) finetuning.

Per-bin cross entropy of the pLDDT, PDE and PAE logits against targets computed
from the mini-rollout structure and the ground truth (MiniWorld
``loss/confidence.py`` + ``models/confidence/client.py::loss_fn``, ported to the
dense ``[res, atoms_per_token]`` layout).

All targets are built under ``no_grad``; the bins are the head's own
(``distance_breaks`` for PDE, ``pae_breaks`` for PAE, 50 uniform lDDT bins for
pLDDT), so a target bin is the bin the head's expected value is read from.

  * pLDDT: per-atom lDDT of the rollout structure against the ground truth (AF3
    section 4.3.1: nucleic neighbours within 30 A, others 15 A; a ligand atom
    counts polymer neighbours only), over the real atoms only.
  * PDE: ``|d_ij(pred) - d_ij(gt)|`` between the token representative
    (pseudo-beta) atoms.
  * PAE: the frame-aligned error of the representative atoms, in the per-token
    frames (N, CA, C for protein; C1', C3', C4' for nucleic acids; a token without
    the three atoms, or whose frame is near colinear, has no frame and drops out of
    the loss).
The experimentally-resolved head has no MiniWorld counterpart: weight 0 in the
MiniWorld recipe (not computed, head frozen); the ``protenix`` recipe
(``train/recipes.py``) trains it with Protenix's ``ExperimentallyResolvedLoss``
(``resolved_loss_value``) and builds the pLDDT target the way Protenix does
(``protenix_targets``: neighbours = Calpha / C1' of polymer tokens, ``<``
thresholds, an atom without neighbours stays in the loss at bin 0).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from foldforge.data.constants import atom_types, residue_names
from foldforge.utils import atom_layout

if TYPE_CHECKING:
    from foldforge.modules.dense.head import ConfidenceHead

PROTEIN_FRAME = (atom_types.N, atom_types.CA, atom_types.C)
NUCLEIC_FRAME = (atom_types.C1PRIME, atom_types.C3PRIME, atom_types.C4PRIME)

#: A representative atom is present when its gathered presence exceeds this.
_PRESENT_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _frame_slot_table() -> tuple[list[tuple[int, int, int]], list[bool]]:
    """Per ``aatype`` index the dense slots of its three frame atoms.

    Also returns whether it has a frame.
    """
    slots, valid = [], []
    for name in residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP:
        order = atom_types.DENSE_ATOM_ORDER.get(name)
        found = None
        if order:
            for names in (PROTEIN_FRAME, NUCLEIC_FRAME):
                if all(n in order for n in names):
                    found = tuple(order[n] for n in names)
                    break
        slots.append(found or (0, 0, 0))
        valid.append(found is not None)
    return slots, valid


@lru_cache(maxsize=8)
def _frame_tables_on(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frame table on ``device``, built once.

    A ``torch.tensor(list, device=cuda)`` per call is a host->device copy that
    synchronises.
    """
    slots, valid = _frame_slot_table()
    return (
        torch.tensor(slots, dtype=torch.long, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )


@lru_cache(maxsize=1)
def _plddt_m_slot_table() -> tuple[list[int], list[bool]]:
    """Per ``aatype`` index the dense slot of its pLDDT neighbour atom.

    The neighbour atom is the Calpha of a protein residue, the C1' of a
    nucleotide; also returns whether it has one.
    """
    slots, valid = [], []
    for name in residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP:
        order = atom_types.DENSE_ATOM_ORDER.get(name) or {}
        found = next(
            (order[a] for a in (atom_types.CA, atom_types.C1PRIME) if a in order), None
        )
        slots.append(0 if found is None else found)
        valid.append(found is not None)
    return slots, valid


@lru_cache(maxsize=8)
def _m_tables_on(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    slots, valid = _plddt_m_slot_table()
    return torch.tensor(slots, dtype=torch.long, device=device), torch.tensor(
        valid, dtype=torch.bool, device=device
    )


def _error_breaks(
    breaks: torch.Tensor,
    protenix: bool,  # noqa: FBT001 - positional flag of the existing call sites
) -> torch.Tensor:
    """Return the bin edges a PDE / PAE error is bucketised with.

    MiniWorld: the head's own ``breaks`` (bin ``k`` = ``(b[k-1], b[k]]``, so an
    error in ``(0, 0.5]`` is bin 1). Protenix (protenix/model/loss.py:693-725 PDE,
    :916-938 PAE): the label is ``#(boundaries < error) - 1`` clamped to
    ``[0, 63]`` over ``linspace(0, 32, 65)``, i.e. bin ``k`` =
    ``(0.5 k, 0.5 (k + 1)]``: the edges shifted up by one step (``b + step``). That
    is also how the head's bin centres (``b[k] + step / 2``) read the logits, so
    the Protenix labels are the ones the released weights were trained on.
    """
    breaks = breaks.float()
    return breaks + (breaks[1] - breaks[0]) if protenix else breaks


def frame_slots(
    aatype: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(slots [res, 3], has_frame [res])`` of the tokens.

    An ``aatype`` outside the polymer alphabet (ligand) has no frame.
    """
    slots, _valid = _frame_slot_table()
    table, ok = _frame_tables_on(aatype.device)
    index = aatype.long().clamp(0, len(slots) - 1)
    in_range = (aatype >= 0) & (aatype < len(slots))
    return table[index], ok[index] & in_range


def per_atom_lddt(
    pred: torch.Tensor,
    gt: torch.Tensor,
    is_nuc: torch.Tensor,
    is_ligand: torch.Tensor,
    max_distance: float = 15.0,
    nuc_max_distance: float = 30.0,
    distance_bins: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    atom_valid: torch.Tensor | None = None,
    neighbour_ok: torch.Tensor | None = None,
    protenix: bool = False,  # noqa: FBT001, FBT002 - public signature
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-atom lDDT ``[N]`` and the validity ``[N]`` of ``N`` real atoms.

    Valid = at least one neighbour; ``pred`` / ``gt`` are ``[N, 3]``.

    ``neighbour_ok`` ``[N]`` (bool, optional): only these atoms may be the
    neighbour ``m`` of an atom ``l`` (Protenix ``calculate_atom_bespoke_lddt``:
    ``m`` = the Calpha / C1' representative of a standard polymer residue,
    ``plddt_m_rep_atom_mask``; protenix/data/core/parser.py:2878-2899).
    ``protenix``: Protenix's ``calculate_atom_bespoke_lddt`` + ``bins_from_lddt``
    arithmetic (loss.py:1212-1280, 1324-1349): thresholds compare with ``<``
    instead of ``<=``, the self pair is removed by an explicit diagonal mask
    (``gt_dist > 0`` is not enough: ``torch.cdist`` of more than 25 points goes
    through a matrix product and its diagonal is not exactly 0), and the value is
    ``(sum of the four indicator counts) * 0.25 / (neighbours + 1e-6)``.

    ``atom_valid`` ``[N]`` (bool, optional): the fixed-length path -- ``N`` is a
    padded length and the ``False`` entries are padding, which is nobody's
    neighbour (the pair mask drops it), so the real atoms' values equal the
    unpadded call's; a padded entry's own output is meaningless (its validity is
    ``False``).
    """
    pred_dist = torch.cdist(pred, pred)
    gt_dist = torch.cdist(gt, gt)
    # python-float scalars (a ``new_tensor`` is a host->device copy, a sync);
    # ``where`` of two scalars is float32 like ``gt_dist``
    radius = torch.where(is_nuc, nuc_max_distance, max_distance).to(gt_dist.dtype)
    pair = (gt_dist > 0.0) & (gt_dist < radius[None, :])
    pair &= ~(is_ligand[:, None] & is_ligand[None, :])
    if atom_valid is not None:
        pair &= atom_valid[:, None] & atom_valid[None, :]
    if neighbour_ok is not None:
        pair &= neighbour_ok[None, :]
    if protenix:
        pair &= ~torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
    delta = (pred_dist - gt_dist).abs()
    # one comparison per bin (python-float bins: no device constant), the same
    # ``[N, 4]`` integer counts as the broadcast ``delta[..., None] <= bins``
    kept = torch.stack(
        [
            (((delta < b) if protenix else (delta <= b)) & pair).sum(dim=-1)
            for b in distance_bins
        ],
        dim=-1,
    )
    if protenix:
        weight = pair.sum(dim=-1).float()
        # Protenix keeps an atom without neighbours (lDDT 0 -> bin 0)
        return kept.sum(dim=-1).float() * 0.25 / (weight + 1e-6), torch.ones_like(
            weight, dtype=torch.bool
        )
    frac = kept.float() / (pair.sum(dim=-1, keepdim=True).float() + 1e-8)
    return frac.mean(dim=-1), pair.sum(dim=-1) > 0


def token_frames(
    atoms: torch.Tensor, frame_ok: torch.Tensor, colinear_cos: float = 0.9063
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rigid frames ``(R [res, 3, 3], t [res, 3], valid [res])``.

    Built from the three frame atoms ``[res, 3, 3]`` (Gram-Schmidt; centre = atom
    1; near-colinear atoms (angle below 25 degrees) invalidate the frame).
    """
    a0, a1, a2 = atoms[:, 0], atoms[:, 1], atoms[:, 2]
    v1, v2 = a2 - a1, a0 - a1
    e1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-8)
    v2n = v2 / (v2.norm(dim=-1, keepdim=True) + 1e-8)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / (u2.norm(dim=-1, keepdim=True) + 1e-8)
    e3 = torch.cross(e1, e2, dim=-1)
    rot = torch.stack([e1, e2, e3], dim=-1)
    colinear = (e1 * v2n).sum(-1).abs() > colinear_cos
    return rot, a1, frame_ok & ~colinear


def _local(rot: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Express representative atom ``j`` in the frame of token ``i``.

    ``R_i^T (x_j - t_i)`` -> ``[res_i, res_j, 3]`` (``rot`` columns are the basis
    vectors, so component ``k`` is ``sum_c rot[i, c, k] * d[c]``; MiniWorld's
    ``pae_target_bins`` contracts the other axis of ``rot``, i.e. applies ``R``
    instead of ``R^T`` -- not carried over).
    """
    return torch.einsum("ick,ijc->ijk", rot, x[None, :, :] - t[:, None, :])


@torch.no_grad()
def confidence_targets(  # noqa: PLR0915 - one linear target build
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    atom_mask: torch.Tensor,
    token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
    token_mask: torch.Tensor,
    aatype: torch.Tensor,
    is_nuc: torch.Tensor,
    is_ligand: torch.Tensor,
    head: ConfidenceHead,
    present_index: torch.Tensor | None = None,
    present_valid: torch.Tensor | None = None,
    real_mask: torch.Tensor | None = None,
    with_pae: bool = True,  # noqa: FBT001, FBT002 - public signature
    protenix_targets: bool = False,  # noqa: FBT001, FBT002 - public signature
) -> dict[str, tuple[torch.Tensor, torch.Tensor] | None]:
    """Return ``{"plddt" | "pde" | "pae": (target bins, mask)}``.

    ``x_pred`` / ``x_gt`` ``[res, atoms, 3]``, ``atom_mask`` ``[res, atoms]``
    (ground-truth atoms present), token tensors ``[res]``.

    ``real_mask`` ``[res, atoms]`` (bool; every atom that exists in the layout,
    resolved or not = ``pred_dense_atom_mask``): adds
    ``"resolved": (is_resolved class 0 / 1, real_mask)`` for the
    experimentally-resolved loss (Protenix ``ExperimentallyResolvedLoss``: the
    class of an atom is ``coordinate_mask``, taken over every atom of the crop).
    ``with_pae=False`` (PAE weight 0): the PAE target is not built
    (``"pae": None``).

    ``present_index`` / ``present_valid`` (both or neither;
    ``TrainBatch.present_index`` / ``present_valid``): the flat indices of the true
    entries of ``atom_mask`` padded to a fixed length, and their validity. Then the
    pLDDT part has a static shape (no ``nonzero``, no host sync); the targets equal
    the ``nonzero`` path's (padding is nobody's neighbour, the scatter back to the
    dense layout skips it).
    """
    res, atoms, _ = x_pred.shape
    x_pred, x_gt = x_pred.float(), x_gt.float()
    present = atom_mask.bool()

    def m_atoms(index: torch.Tensor) -> torch.Tensor | None:
        """Return which of the flat atoms ``index`` may be a neighbour ``m``.

        ``protenix_targets`` only (Calpha / C1' of a polymer token), else ``None``.
        """
        if not protenix_targets:
            return None
        table, ok = _m_tables_on(aatype.device)
        token = aatype.long().clamp(0, table.shape[0] - 1)
        usable = (
            ok[token] & (aatype >= 0) & (aatype < table.shape[0]) & ~is_ligand.bool()
        )
        owner = index // atoms
        return usable[owner] & ((index % atoms) == table[token][owner])

    # pLDDT over the real atoms only (a dense crop has many empty slots).
    if present_index is None:
        flat = present.reshape(-1)
        index = flat.nonzero(as_tuple=True)[0]
        token_of_atom = index // atoms
        lddt, valid = per_atom_lddt(
            x_pred.reshape(-1, 3)[index],
            x_gt.reshape(-1, 3)[index],
            is_nuc.bool()[token_of_atom],
            is_ligand.bool()[token_of_atom],
            neighbour_ok=m_atoms(index),
            protenix=protenix_targets,
        )
        # Protenix keeps an atom without neighbours (weight 0 -> lDDT 0 -> bin 0)
        if protenix_targets:
            valid = torch.ones_like(valid)
        lddt_dense = torch.zeros(res * atoms, device=x_pred.device)
        valid_dense = torch.zeros(res * atoms, dtype=torch.bool, device=x_pred.device)
        lddt_dense[index] = lddt
        valid_dense[index] = valid
    else:
        # fixed length: padding reads atom 0 (masked out of every pair) and writes
        # the spare slot ``res * atoms``
        index = present_index
        token_of_atom = index // atoms
        lddt, valid = per_atom_lddt(
            x_pred.reshape(-1, 3)[index],
            x_gt.reshape(-1, 3)[index],
            is_nuc.bool()[token_of_atom],
            is_ligand.bool()[token_of_atom],
            atom_valid=present_valid,
            neighbour_ok=m_atoms(index),
            protenix=protenix_targets,
        )
        if protenix_targets:
            valid = present_valid.clone()
        target = torch.where(present_valid, index, res * atoms)
        lddt_dense = torch.zeros(res * atoms + 1, device=x_pred.device)
        valid_dense = torch.zeros(
            res * atoms + 1, dtype=torch.bool, device=x_pred.device
        )
        lddt_dense[target] = lddt
        valid_dense[target] = valid
        lddt_dense, valid_dense = lddt_dense[:-1], valid_dense[:-1]
    num_bins = head.num_plddt_bins
    if protenix_targets:
        # Protenix PLDDTLoss.bins_from_lddt (loss.py:1324-1349):
        # ``#(lddt > linspace(0, 1, 51)) clamped to [1, 50]``, minus one -> bin k =
        # (0.02 k, 0.02 (k + 1)]
        boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=lddt_dense.device)
        plddt_bins = (
            (lddt_dense[:, None] > boundaries).sum(dim=-1).clamp(1, num_bins) - 1
        ).reshape(res, atoms)
    else:
        plddt_bins = (
            (lddt_dense.clamp(0.0, 1.0 - 1e-6) * num_bins).long().reshape(res, atoms)
        )

    # Representative atoms.
    beta_pred = atom_layout.convert(
        token_atoms_to_pseudo_beta, x_pred, layout_axes=(-3, -2)
    )
    beta_gt = atom_layout.convert(
        token_atoms_to_pseudo_beta, x_gt, layout_axes=(-3, -2)
    )
    beta_ok = (
        atom_layout.convert(
            token_atoms_to_pseudo_beta, present.float(), layout_axes=(-2, -1)
        )
        > _PRESENT_THRESHOLD
    )
    tok_ok = beta_ok & (token_mask > 0)
    pair_ok = tok_ok[:, None] & tok_ok[None, :]

    pde_err = (torch.cdist(beta_pred, beta_pred) - torch.cdist(beta_gt, beta_gt)).abs()
    pde_bins = torch.bucketize(
        pde_err, _error_breaks(head.distance_breaks, protenix_targets)
    )

    targets: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {
        "plddt": (plddt_bins, valid_dense.reshape(res, atoms)),
        "pde": (pde_bins, pair_ok),
        "pae": None,
    }
    if real_mask is not None:
        targets["resolved"] = (present.long(), real_mask.bool())
    if not with_pae:
        return targets

    # Frame-aligned error.
    slots, has_frame = frame_slots(aatype)
    rows = torch.arange(res, device=x_pred.device)[:, None]
    pred_frame_atoms = x_pred[rows, slots]
    gt_frame_atoms = x_gt[rows, slots]
    observed = present[rows, slots].all(dim=-1)
    r_pred, t_pred, ok_pred = token_frames(pred_frame_atoms, has_frame)
    r_gt, t_gt, ok_gt = token_frames(gt_frame_atoms, has_frame & observed)
    pae_err = (_local(r_pred, t_pred, beta_pred) - _local(r_gt, t_gt, beta_gt)).norm(
        dim=-1
    )
    pae_bins = torch.bucketize(
        pae_err, _error_breaks(head.pae_breaks, protenix_targets)
    )
    pae_mask = (ok_pred & ok_gt & (token_mask > 0))[:, None] & tok_ok[None, :]
    targets["pae"] = (pae_bins, pae_mask)
    return targets


def masked_ce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean cross entropy over the masked positions.

    ``logits`` ``[..., C]``, ``target`` / ``mask`` ``[...]``.
    """
    classes = logits.shape[-1]
    ce = F.cross_entropy(
        logits.float().reshape(-1, classes),
        target.reshape(-1).clamp(0, classes - 1),
        reduction="none",
    )
    weight = mask.reshape(-1).float()
    return (ce * weight).sum() / (weight.sum() + 1e-8)


def confidence_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    plddt_loss: float,
    pde_loss: float,
    pae_loss: float,
    resolved_loss: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of the per-bin cross entropies of the head's logits.

    The logits are ``outputs["plddt_logits" | "pde_logits" | "pae_logits" |
    "resolved_logits"]`` (the head run with ``save_logits``). A term whose weight
    is exactly 0 is not computed (reported as 0.0), like a Protenix loss with
    ``weight * loss`` = 0 contributes nothing. Returns ``(total, {name: loss})``:
    ``plddt`` / ``pde`` / ``pae`` always, ``resolved`` only when its weight is
    nonzero.
    """
    zero = outputs["plddt_logits"].new_zeros(()).float()
    parts = {
        "plddt": masked_ce(outputs["plddt_logits"], *targets["plddt"])
        if plddt_loss != 0.0
        else zero,
        "pde": masked_ce(outputs["pde_logits"], *targets["pde"])
        if pde_loss != 0.0
        else zero,
        "pae": masked_ce(outputs["pae_logits"], *targets["pae"])
        if pae_loss != 0.0
        else zero,
    }
    if resolved_loss != 0.0:
        parts["resolved"] = resolved_loss_value(
            outputs["resolved_logits"], *targets["resolved"]
        )
    total = zero
    for name, weight in (
        ("plddt", plddt_loss),
        ("pde", pde_loss),
        ("pae", pae_loss),
        ("resolved", resolved_loss),
    ):
        if weight != 0.0:
            total = total + weight * parts[name]
    return total, parts


def resolved_loss_value(
    logits: torch.Tensor, resolved: torch.Tensor, real_mask: torch.Tensor
) -> torch.Tensor:
    """Experimentally-resolved loss of one sample.

    Mean over the atoms of ``real_mask`` of the 2-class cross entropy of
    ``logits`` ``[res, atoms, 2]`` against ``resolved`` (the ground-truth atom is
    present). Protenix ``ExperimentallyResolvedLoss.forward``
    (protenix/model/loss.py:1026-1060).
    """
    return masked_ce(logits, resolved, real_mask)
