"""CPU tests of ``foldforge.train.bucket_batch``.

Padding / mask logic on small fake examples, the signature function, and the
atom-attention gather tables against AF3's own ``AtomCrossAtt.compute_features`` (the
rule the padded windows must follow).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from foldforge.train.bucket_batch import (
    BucketOverflow,
    BucketSpec,
    bucket_example,
    count_signatures,
    shape_signature,
    signature_diff,
)
from foldforge.train.data import TrainBatch

features = pytest.importorskip("alphafold3.model.features")
atom_layout = pytest.importorskip("alphafold3.model.atom_layout.atom_layout")

TABLES = (
    "token_atoms_to_queries",
    "tokens_to_queries",
    "tokens_to_keys",
    "queries_to_keys",
    "queries_to_token_atoms",
)


def make_layout(atoms_per_token: list[int]):
    """A token-atom layout [tokens, 24] with unique (chain, residue, atom) uids.

    ``atoms_per_token[t]`` real atoms in token ``t``.
    """
    t = len(atoms_per_token)
    name = np.full((t, 24), "", dtype=object)
    res = np.zeros((t, 24), dtype=int)
    chain = np.full((t, 24), "A", dtype=object)
    for i, k in enumerate(atoms_per_token):
        name[i, :k] = [f"X{j}" for j in range(k)]
        res[i, :] = i + 1
    other = np.full((t, 24), "", dtype=object)
    return atom_layout.AtomLayout(
        atom_name=name,
        res_id=res,
        chain_id=chain,
        atom_element=other.copy(),
        res_name=other.copy(),
        chain_type=other.copy(),
    )


def af3_tables(layout, tokens: int, atoms: int) -> dict[str, torch.Tensor]:
    result = features.AtomCrossAtt.compute_features(
        all_token_atoms_layout=layout,
        queries_subset_size=32,
        keys_subset_size=128,
        padding_shapes=features.PaddingShapes(
            num_tokens=tokens,
            msa_size=0,
            num_chains=0,
            num_templates=0,
            num_atoms=atoms,
        ),
    ).as_data_dict()
    return {k: torch.as_tensor(np.asarray(v)) for k, v in result.items()}


def counts(n_tokens: int, n_atoms: int) -> list[int]:
    """``n_tokens`` tokens of 1..24 atoms that sum to exactly ``n_atoms``."""
    base = [n_atoms // n_tokens] * n_tokens
    for i in range(n_atoms - sum(base)):
        base[i] += 1
    assert max(base) <= 24
    return base


def fake_example(
    t_real: int,
    n_atoms: int,
    t_bucket: int,
    a_bucket: int,
    msa_rows: int,
    templates: int = 4,
    name: str = "fake",
    msa_valid: int | None = None,
) -> TrainBatch:
    layout = make_layout(counts(t_real, n_atoms))
    f = af3_tables(layout, t_bucket, a_bucket)
    dense = torch.zeros(t_bucket, 24, dtype=torch.bool)
    for t, k in enumerate(counts(t_real, n_atoms)):
        dense[t, :k] = True
    seq_mask = torch.zeros(t_bucket, dtype=torch.bool)
    seq_mask[:t_real] = True
    valid = msa_rows if msa_valid is None else msa_valid
    msa_mask = torch.zeros(msa_rows, t_bucket, dtype=torch.bool)
    msa_mask[:valid, :t_real] = True
    g = torch.Generator().manual_seed(1)
    f.update(
        {
            "seq_mask": seq_mask,
            "pred_dense_atom_mask": dense,
            "aatype": torch.randint(0, 20, (t_bucket,), generator=g, dtype=torch.int32)
            * seq_mask,
            "msa": (
                torch.randint(0, 20, (msa_rows, t_bucket), generator=g) * msa_mask
            ).to(torch.int8),
            "msa_mask": msa_mask,
            "deletion_matrix": torch.zeros(msa_rows, t_bucket, dtype=torch.int8),
            "profile": torch.rand(t_bucket, 31, generator=g) * seq_mask[:, None],
            "num_alignments": torch.tensor([valid], dtype=torch.int32),
            "template_aatype": torch.zeros(templates, t_bucket, dtype=torch.int32),
            "template_atom_mask": torch.zeros(
                templates, t_bucket, 24, dtype=torch.bool
            ),
            "template_atom_positions": torch.zeros(templates, t_bucket, 24, 3),
            "ref_pos": torch.rand(t_bucket, 24, 3, generator=g, dtype=torch.float64),
            "tokens_to_ligand_ligand_bonds:gather_idxs": torch.zeros(
                t_bucket * 10, 2, dtype=torch.int64
            ),
            "tokens_to_ligand_ligand_bonds:gather_mask": torch.zeros(
                t_bucket * 10, 2, dtype=torch.bool
            ),
            "ligand_ligand_bond_order": torch.zeros(t_bucket * 10, dtype=torch.int32),
            "tokens_to_polymer_ligand_bonds:gather_idxs": torch.zeros(
                t_bucket, 2, dtype=torch.int64
            ),
            "tokens_to_polymer_ligand_bonds:input_shape": torch.tensor([t_real]),
            "token_atoms_to_polymer_ligand_bonds:input_shape": torch.tensor(
                [t_bucket, 24]
            ),
        }
    )
    return TrainBatch(
        features=f,
        atom_positions=torch.rand(t_bucket, 24, 3, generator=g) * dense[..., None],
        name=name,
    )


# (tokens, atoms) of the real content; old bucket; new bucket
CASES = [
    pytest.param(20, 160, (24, 192), (32, 288), id="grow tokens and atoms"),
    pytest.param(20, 160, (32, 288), (24, 192), id="shrink the padding of both"),
    pytest.param(20, 160, (24, 160), (32, 320), id="atoms exactly fill the old bucket"),
    pytest.param(24, 190, (24, 192), (24, 192), id="same bucket"),
    pytest.param(32, 300, (32, 320), (32, 640), id="no pad token, atoms only"),
    pytest.param(32, 300, (32, 640), (32, 320), id="no pad token, fewer atoms"),
    pytest.param(30, 320, (32, 320), (40, 480), id="fill, then both grow"),
]


@pytest.mark.parametrize(("t_real", "n_atoms", "old", "new"), CASES)
def test_gather_tables_equal_af3_padding(t_real, n_atoms, old, new):
    example = fake_example(t_real, n_atoms, *old, msa_rows=8)
    spec = BucketSpec(tokens=new[0], atoms=new[1], msa_pool_rows=8)
    out = bucket_example(example, spec)
    layout = make_layout(counts(t_real, n_atoms))
    want = af3_tables(layout, *new)
    for table in TABLES:
        for part in ("gather_idxs", "gather_mask", "input_shape"):
            key = f"{table}:{part}"
            assert torch.equal(out.features[key], want[key]), key
            assert out.features[key].dtype == want[key].dtype, key


def test_every_example_gets_one_signature():
    spec = BucketSpec(tokens=32, atoms=640, msa_pool_rows=64)
    pool = [
        fake_example(20, 160, 24, 192, msa_rows=8, name="a"),
        fake_example(30, 300, 32, 320, msa_rows=16, name="b", templates=4),
        fake_example(32, 600, 32, 640, msa_rows=64, name="c"),
        fake_example(10, 140, 16, 160, msa_rows=4, name="d", templates=2),
    ]
    assert len(count_signatures(pool)) == 4
    bucketed = [bucket_example(e, spec) for e in pool]
    signatures = count_signatures(bucketed)
    assert len(signatures) == 1
    shapes = dict(next(iter(signatures)))
    assert shapes["features.msa"][0] == (64, 32)
    assert shapes["features.queries_to_keys:gather_idxs"][0] == (20, 128)
    assert shapes["features.template_atom_mask"][0] == (4, 32, 24)
    assert shapes["atom_positions"][0] == (32, 24, 3)
    assert shapes["atom_mask"][0] == (32, 24)
    assert signature_diff(pool[0], bucketed[0])


def test_padding_is_masked_and_real_content_unchanged():
    ex = fake_example(20, 160, 24, 192, msa_rows=8, templates=2)
    spec = BucketSpec(tokens=32, atoms=288, msa_pool_rows=16, templates=4)
    out = bucket_example(ex, spec)
    f, g = ex.features, out.features
    assert int(g["seq_mask"].sum()) == 20
    assert not g["seq_mask"][20:].any()
    assert not g["pred_dense_atom_mask"][20:].any()
    assert torch.equal(g["pred_dense_atom_mask"][:24], f["pred_dense_atom_mask"])
    assert torch.equal(g["msa"][:8, :24], f["msa"])
    assert not g["msa_mask"][8:].any()
    assert not g["msa_mask"][:, 24:].any()
    assert torch.equal(g["msa_mask"][:8, :24], f["msa_mask"])
    assert not g["template_atom_mask"][2:].any()
    assert torch.equal(g["profile"][:24], f["profile"])
    assert not g["profile"][24:].any()
    assert torch.equal(out.atom_positions[:24], ex.atom_positions)
    assert not out.atom_positions[24:].any()
    # None -> the model's own mask
    assert torch.equal(out.atom_mask, g["pred_dense_atom_mask"])
    assert out.bond_pairs is None
    # dtypes never change
    assert {k: v.dtype for k, v in f.items()} == {k: v.dtype for k, v in g.items()}
    # already at the bucket shape: the same tensors (bit-identical)
    again = bucket_example(out, spec)
    assert all(
        again.features[k] is out.features[k]
        or torch.equal(again.features[k], out.features[k])
        for k in out.features
    )
    assert count_signatures([out, again]) == {shape_signature(out): 2}


def test_bond_pairs_padded_with_minus_one():
    ex = fake_example(20, 160, 24, 192, msa_rows=8)
    ex.bond_pairs = torch.tensor([[0, 1, 2, 3], [4, 0, 5, 1]])
    out = bucket_example(
        ex, BucketSpec(tokens=32, atoms=288, msa_pool_rows=8, bond_pairs=6)
    )
    assert out.bond_pairs.shape == (6, 4)
    assert torch.equal(out.bond_pairs[:2], ex.bond_pairs)
    assert (out.bond_pairs[2:] == -1).all()
    assert bucket_example(
        fake_example(20, 160, 24, 192, msa_rows=8),
        BucketSpec(tokens=32, atoms=288, msa_pool_rows=8, bond_pairs=6),
    ).bond_pairs.shape == (6, 4)
    with pytest.raises(BucketOverflow):
        bucket_example(
            ex, BucketSpec(tokens=32, atoms=288, msa_pool_rows=8, bond_pairs=1)
        )


def test_overflow_is_rejected():
    ex = fake_example(30, 600, 32, 640, msa_rows=8)
    with pytest.raises(BucketOverflow, match="atoms"):
        bucket_example(ex, BucketSpec(tokens=32, atoms=512, msa_pool_rows=8))
    with pytest.raises(BucketOverflow, match="tokens"):
        bucket_example(ex, BucketSpec(tokens=24, atoms=640, msa_pool_rows=8))


def test_msa_pool_above_the_bucket_keeps_the_query_and_valid_rows():
    ex = fake_example(20, 160, 24, 192, msa_rows=40)
    spec = BucketSpec(tokens=24, atoms=192, msa_pool_rows=16)
    a = bucket_example(ex, spec)
    b = bucket_example(ex, spec)
    assert a.features["msa"].shape == (16, 24)
    assert a.features["msa_mask"].any(-1).all()  # 16 valid rows, no padding row
    # the query stays on top
    assert torch.equal(a.features["msa"][0], ex.features["msa"][0])
    assert torch.equal(a.features["msa"], b.features["msa"])  # deterministic
    # fewer valid rows than the bucket: only empty rows are cut, valid ones keep their
    # order
    ex2 = fake_example(20, 160, 24, 192, msa_rows=40, msa_valid=5)
    c = bucket_example(ex2, spec)
    assert torch.equal(c.features["msa"][:5], ex2.features["msa"][:5])
    assert int(c.features["msa_mask"].any(-1).sum()) == 5


def test_real_tokens_must_be_in_front():
    ex = fake_example(20, 160, 24, 192, msa_rows=8)
    ex.features["seq_mask"][3] = False
    with pytest.raises(ValueError, match="in front"):
        bucket_example(ex, BucketSpec(tokens=32, atoms=288, msa_pool_rows=8))
