"""One fixed tensor shape for every training example (shape bucketing).

A CUDA graph of the micro step is then keyed on ONE signature.

Why. FoldJet's featurisation sizes three things by the example: the token axis
(multiples of 128), the flat-atom axis (the atom ladder 1024 / 2048 / 4096 / ...: it
is the number of 32-atom query windows of the atom-attention gather tables) and the
raw MSA pool (1024 .. 16384 rows, before the per-recycle draw of ``num_msa`` = 1024
rows). A step that is captured in a CUDA graph is keyed on input shapes; on the
default stand-in pool (7-8 FoldBench targets of 285-384 tokens) these three axes give
5 different feature signatures. MiniWorld's trainer pads every batch to ONE bucket
(``bucketed_collate`` with ``bucket_token_multiple: 384``,
``bucket_atom_multiple: 4096``: a single bucket = the crop of
``configs/miniworld/data/H100_default.yaml``, ``crop.max_tokens 384`` /
``max_atoms 4096``); this module does the same to a FoldJet
:class:`~foldjet.train.data.TrainBatch`.

What the padding is. Every padded element is MASKED, so the forward and the loss of
the padded example are those of the original:

* tokens: zero rows with ``seq_mask`` / ``pred_dense_atom_mask`` False (the
  featurisation's own token padding, extended or trimmed);
* atoms: the atom-attention gather tables (``token_atoms_to_queries`` ...
  ``queries_to_token_atoms``) get more (or fewer) empty 32-atom query windows, built
  by the rule AF3's ``AtomCrossAtt.compute_features`` uses for padded windows (checked
  against it in ``tests/train/test_bucket_batch.py``); the dense tensors
  ``[res, 24, ...]`` do not change with the atom bucket at all;
* MSA pool: zero rows with ``msa_mask`` False up to ``msa_pool_rows`` (the per-recycle
  draw, ``Evoformer.sample_msa``, ranks masked rows below every valid row, so it picks
  the same valid rows; a pool above ``msa_pool_rows`` keeps the query and a
  deterministic sample of the valid rows);
* templates: zero templates (``template_atom_mask`` False) up to ``templates``; more
  are cut;
* ground truth: ``atom_positions`` zeros, ``atom_mask`` False, ``bond_pairs`` ``-1``
  rows (the loss reads ``>= 0`` as valid).

An example whose REAL content does not fit (more atoms than ``atoms``, more tokens than
``tokens``) raises :class:`BucketOverflow`: cropping a featurised example would need
the featuriser (MiniWorld crops at the loader, ``crop.max_atoms``); the loader must
hand over examples that fit.

One honest difference: the MSA draw is a gumbel noise vector over the POOL rows, so a
different pool length consumes the torch RNG differently (another valid draw of the
same distribution; unchanged when the pool length is unchanged).
"""

from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from foldjet.train.data import TrainBatch

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

ATOMS_PER_TOKEN = 24  # the dense layout [res, 24]
QUERY_BLOCK = 32  # atoms per query window of the atom-attention gather tables

#: Keys whose token axis is not axis 0 (name -> axis). Every other key is recognised by
#: its first dimension (see :func:`_token_axis`).
_TOKEN_AXIS = {
    "msa": 1,
    "msa_mask": 1,
    "deletion_matrix": 1,
    "template_aatype": 1,
    "template_atom_positions": 1,
    "template_atom_mask": 1,
}
_MSA_ROWS = ("msa", "msa_mask", "deletion_matrix")
_TEMPLATES = ("template_aatype", "template_atom_positions", "template_atom_mask")
#: The atom-attention tables and the keys derived from the token axis by a fixed factor.
_WINDOW_TABLES = (
    "token_atoms_to_queries",
    "tokens_to_queries",
    "tokens_to_keys",
    "queries_to_keys",
    "queries_to_token_atoms",
)
_LIGAND_BONDS = (
    "tokens_to_ligand_ligand_bonds:gather_idxs",
    "tokens_to_ligand_ligand_bonds:gather_mask",
    "ligand_ligand_bond_order",
)
_LIGAND_BONDS_PER_TOKEN = 10  # first dimension of those keys = 10 x tokens


class BucketOverflow(ValueError):  # noqa: N818 - public exception name
    """The example holds more real tokens / atoms than the bucket."""


@dataclass(frozen=True)
class BucketSpec:
    """The ONE shape of every example of a run.

    MiniWorld's recipe crop: 384 tokens, 4096 atoms.
    """

    tokens: int = 384
    # flat atoms: a multiple of 32 (the number of query windows is atoms / 32)
    atoms: int = 4096
    #: raw MSA rows before the per-recycle draw. 16384 = the largest MSA bucket
    #: FoldJet's featurisation produces (``bucketing.MSA_SHAPES``): nothing is ever
    #: cut; the cost is 3 int8/bool arrays [rows, tokens] = 18.9 MB per example at 384
    #: tokens.
    msa_pool_rows: int = 16384
    templates: int = 4
    #: rows of ``bond_pairs`` ([P, 4], -1 padded); 0 = ``bond_pairs`` is not
    #: materialised (a ``None`` stays ``None``, a tensor keeps its rows)
    bond_pairs: int = 0
    seed: int = 0  # the sample of an MSA pool above ``msa_pool_rows``

    def __post_init__(self) -> None:
        if (
            self.atoms % QUERY_BLOCK
            or min(self.tokens, self.atoms, self.msa_pool_rows, self.templates) < 1
        ):
            message = f"invalid BucketSpec {self}"
            raise ValueError(message)


# ---------------------------------------------------------------------------- signature
def shape_signature(example: TrainBatch | dict[str, torch.Tensor]) -> tuple:
    """Hashable ``(key, shape, dtype)`` of every tensor of an example.

    The tensors of ``features`` plus ``atom_positions`` / ``atom_mask`` /
    ``bond_pairs`` (``None`` kept as such): two examples with the same signature run
    the same graph. A CUDA graph of the step is keyed on exactly this.
    """
    items: dict[str, torch.Tensor | None]
    if isinstance(example, TrainBatch):
        items = {f"features.{k}": v for k, v in example.features.items()}
        items["atom_positions"] = example.atom_positions
        items["atom_mask"] = example.atom_mask
        items["bond_pairs"] = example.bond_pairs
    else:
        items = dict(example)
    return tuple(
        (k, None if v is None else (tuple(v.shape), str(v.dtype)))
        for k, v in sorted(items.items())
    )


def count_signatures(
    examples: Iterable[TrainBatch | dict[str, torch.Tensor]],
) -> Counter:
    """``Counter`` of the distinct :func:`shape_signature` values of ``examples``.

    Its length is the number of graphs a shape-keyed capture would take.
    """
    return Counter(shape_signature(e) for e in examples)


def signature_diff(a: TrainBatch, b: TrainBatch) -> dict[str, tuple]:
    """Keys whose ``(shape, dtype)`` differ between two examples (debugging aid)."""
    sa, sb = dict(shape_signature(a)), dict(shape_signature(b))
    return {
        k: (sa.get(k), sb.get(k))
        for k in sorted(set(sa) | set(sb))
        if sa.get(k) != sb.get(k)
    }


# ------------------------------------------------------------------------------ helpers
def _resize(t: torch.Tensor, axis: int, extent: int, fill: float = 0) -> torch.Tensor:
    """Cut or pad one axis at its end."""
    n = t.shape[axis]
    if n == extent:
        return t
    if n > extent:
        return t.narrow(axis, 0, extent)
    pad = [0, 0] * (t.ndim - axis - 1) + [0, extent - n]
    return F.pad(t, pad, value=fill)


def _token_axis(key: str, t: torch.Tensor, tokens_old: int) -> int | None:
    axis = _TOKEN_AXIS.get(key, 0)
    return axis if t.ndim > axis and t.shape[axis] == tokens_old else None


def real_counts(features: dict[str, torch.Tensor]) -> tuple[int, int]:
    """``(real tokens, real atoms)`` of a featurised example."""
    mask = features["pred_dense_atom_mask"]
    return int(features["seq_mask"].sum()), int(mask.sum())


def _pad_atom_tables(
    f: dict[str, torch.Tensor],
    out: dict[str, torch.Tensor],
    t_real: int,
    n_atoms: int,
    t_old: int,
    t_new: int,
    a_new: int,
) -> None:
    """Rebuild the five atom-attention gather tables from the old ones.

    The new tables are for ``t_new`` tokens and ``a_new`` flat atoms (real entries
    untouched). The values of a PADDED entry follow ``compute_gather_idxs``: the
    padding uid (chain / residue / atom name of a pad slot) is found in the source
    layout at its LAST slot when the source has one, else ``(0, False)``:

    * ``token_atoms_to_queries`` (queries <- token atoms): pad query -> last token-atom
      slot ``t_new * 24 - 1``, True; (0, False) when no pad token;
    * ``tokens_to_queries``: the token of that slot, mask False;
    * ``queries_to_keys`` / ``tokens_to_keys``: a window whose key range runs past the
      last real atom is shifted back to ``[n - 128, n - 1]``; this holds for the last
      two windows of any table with ``atoms >= n``, so every new window copies the
      last old row;
    * ``queries_to_token_atoms`` (token atoms <- queries): the slots of a pad token ->
      last flat query ``a_new - 1`` (a pad query exists iff ``a_new > n``), True.
    """
    del t_old
    nw_new = a_new // QUERY_BLOCK
    tq = "token_atoms_to_queries"
    pad_source = t_real < t_new
    flat_pad_idx = t_new * ATOMS_PER_TOKEN - 1 if pad_source else 0
    flat_query = torch.arange(
        nw_new * QUERY_BLOCK, device=f[tq + ":gather_idxs"].device
    ).view(nw_new, QUERY_BLOCK)
    is_pad = flat_query >= n_atoms  # [nw, 32]

    for name, pad_idx, pad_mask in (
        (tq, flat_pad_idx, pad_source),
        ("tokens_to_queries", flat_pad_idx // ATOMS_PER_TOKEN, False),
    ):
        idx = _resize(f[name + ":gather_idxs"], 0, nw_new)
        mask = _resize(f[name + ":gather_mask"], 0, nw_new)
        out[name + ":gather_idxs"] = torch.where(
            is_pad, torch.full_like(idx, pad_idx), idx
        )
        out[name + ":gather_mask"] = torch.where(
            is_pad, torch.full_like(mask, pad_mask), mask
        )

    for name in ("tokens_to_keys", "queries_to_keys"):
        for part in (":gather_idxs", ":gather_mask"):
            old = f[name + part]
            tail = old[-1:].expand(max(nw_new - old.shape[0], 0), *old.shape[1:])
            out[name + part] = (
                torch.cat([old[:nw_new], tail])
                if nw_new > old.shape[0]
                else old[:nw_new]
            )

    # token atoms <- queries, [tokens, 24]; real tokens' rows are unchanged, the rows of
    # pad tokens are rebuilt
    name = "queries_to_token_atoms"
    idx = _resize(f[name + ":gather_idxs"], 0, t_new)
    mask = _resize(f[name + ":gather_mask"], 0, t_new)
    pad_token = (torch.arange(t_new, device=idx.device) >= t_real)[:, None]
    pad_query = a_new > n_atoms
    out[name + ":gather_idxs"] = torch.where(
        pad_token, torch.full_like(idx, a_new - 1 if pad_query else 0), idx
    )
    out[name + ":gather_mask"] = torch.where(
        pad_token, torch.full_like(mask, pad_query), mask
    )

    # the recorded source shapes follow the tables they describe
    key = tq + ":input_shape"
    out[key] = torch.tensor(
        [t_new, ATOMS_PER_TOKEN], dtype=f[key].dtype, device=f[key].device
    )
    for key in ("tokens_to_queries:input_shape", "tokens_to_keys:input_shape"):
        out[key] = torch.tensor([t_new], dtype=f[key].dtype, device=f[key].device)
    for key in ("queries_to_keys:input_shape", "queries_to_token_atoms:input_shape"):
        out[key] = torch.tensor(
            [nw_new, QUERY_BLOCK], dtype=f[key].dtype, device=f[key].device
        )


def _sample_pool(msa_mask: torch.Tensor, rows: int, seed: int) -> torch.Tensor:
    """Pick ``rows`` rows of a pool with more valid rows than that.

    Returns their indices (sorted, query row 0 first): the query + a seeded sample of
    the rest.
    """
    valid = msa_mask.reshape(msa_mask.shape[0], -1).any(-1).nonzero().squeeze(-1).cpu()
    if valid.numel() <= rows:
        return valid.to(msa_mask.device)
    others = valid[valid != 0]
    gen = torch.Generator().manual_seed(seed)
    keep = others[torch.randperm(others.numel(), generator=gen)[: rows - 1]]
    return torch.cat([valid.new_zeros(1), keep]).sort().values.to(msa_mask.device)


# ------------------------------------------------------------------------ the bucketing
def bucket_example(  # noqa: C901, PLR0912 - one linear pass over the feature groups
    example: TrainBatch, spec: BucketSpec
) -> TrainBatch:
    """Pad ``example`` (or trim its padding) to ``spec``.

    Identical shapes and dtypes for every example, masks keep the loss unchanged.

    Raises :class:`BucketOverflow` when the real tokens / atoms do not fit. Tensors
    that already have the bucket shape are returned as they are (the same objects), so
    an example that the featurisation already padded to the bucket is bit-identical.
    """
    f = example.features
    t_old = f["seq_mask"].shape[0]
    t_real, n_atoms = real_counts(f)
    if t_real > spec.tokens or n_atoms > spec.atoms:
        message = (
            f"{example.name or 'example'}: {t_real} tokens / {n_atoms} atoms do not "
            f"fit the bucket {spec.tokens} / {spec.atoms}"
        )
        raise BucketOverflow(message)
    if not bool(f["seq_mask"][:t_real].all()):
        message = (
            "bucket_example needs the real tokens in front (padding at the end of the "
            "token axis)"
        )
        raise ValueError(message)
    t_new = spec.tokens
    out: dict[str, torch.Tensor] = {}
    atom_keys = {
        f"{n}:{p}"
        for n in _WINDOW_TABLES
        for p in ("gather_idxs", "gather_mask", "input_shape")
    }
    for key, value in f.items():
        if key in atom_keys:
            continue
        if key in _TEMPLATES:
            out[key] = _resize(value, 0, spec.templates)
        elif key in _LIGAND_BONDS:
            out[key] = _resize(value, 0, t_new * _LIGAND_BONDS_PER_TOKEN)
        else:
            out[key] = value
    # token axis
    for key in list(out):
        if key in _LIGAND_BONDS or key.endswith(":input_shape"):
            continue
        axis = _token_axis(key, out[key], t_old)
        if axis is not None:
            out[key] = _resize(out[key], axis, t_new)
    # input_shape metadata that records the padded token axis
    for key in list(out):
        if key.endswith(":input_shape") and key not in atom_keys:
            shape = out[key].tolist()
            if shape[0] == t_old:
                out[key] = torch.tensor(
                    [t_new, *shape[1:]], dtype=out[key].dtype, device=out[key].device
                )
    # MSA pool rows
    if f["msa"].shape[0] != spec.msa_pool_rows:
        index = (
            _sample_pool(
                f["msa_mask"],
                spec.msa_pool_rows,
                spec.seed ^ zlib.crc32(example.name.encode()),
            )
            if (f["msa_mask"].shape[0] > spec.msa_pool_rows)
            else None
        )
        for key in _MSA_ROWS:
            rows = out[key] if index is None else out[key][index]
            out[key] = _resize(rows, 0, spec.msa_pool_rows)
    # atom-attention tables
    _pad_atom_tables(f, out, t_real, n_atoms, t_old, t_new, spec.atoms)

    positions = _resize(example.atom_positions, 0, t_new)
    atom_mask = example.atom_mask
    atom_mask = (
        out["pred_dense_atom_mask"].clone()
        if atom_mask is None
        else _resize(atom_mask, 0, t_new)
    )
    bonds = example.bond_pairs
    if spec.bond_pairs:
        if bonds is None:
            bonds = torch.full(
                (spec.bond_pairs, 4), -1, dtype=torch.int64, device=positions.device
            )
        elif bonds.shape[0] > spec.bond_pairs:
            message = f"{bonds.shape[0]} bond pairs exceed the bucket {spec.bond_pairs}"
            raise BucketOverflow(message)
        else:
            bonds = _resize(bonds, 0, spec.bond_pairs, fill=-1)
    # keep the dtype order of the features (and their key order) so that the signature
    # is comparable
    return replace(
        example,
        features={k: out[k] for k in f},
        atom_positions=positions,
        atom_mask=atom_mask,
        bond_pairs=bonds,
    )


def bucket_source(
    source: Iterable[TrainBatch], spec: BucketSpec
) -> Iterator[TrainBatch]:
    """Bucket a stream of examples (a ``--batch-source`` iterator) to ``spec``."""
    for example in source:
        yield bucket_example(example, spec)
