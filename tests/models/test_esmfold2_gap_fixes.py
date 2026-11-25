"""The 5I28 ESMFold2 forward fixes.

The relative encoding built in its target dtype, the confidence head's per-fold
re-embedding terms built once. Both are exact rewrites: the same values, the same
operations in the same order, fewer launches and no int64 one-hot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from foldjet.data.features import dense as features
from foldjet.modules.dense import featurization, head
from foldjet.modules.dense.head import ConfidenceHead, ConfidenceReembedding
from foldjet.modules.dense.spec import SPECS


def _tokens(n: int = 12, chains: int = 3) -> features.TokenFeatures:
    chain = torch.arange(n) * chains // n
    return features.TokenFeatures(
        residue_index=torch.arange(n) % (n // chains),
        token_index=torch.arange(n),
        asym_id=chain,
        entity_id=torch.where(chain == 2, 2, 1),
        sym_id=chain % 2,
        aatype=torch.zeros(n, dtype=torch.long),
        is_protein=torch.ones(n),
        is_rna=torch.zeros(n),
        is_dna=torch.zeros(n),
        is_ligand=torch.zeros(n),
        is_water=torch.zeros(n),
        is_nonstandard_polymer_chain=torch.zeros(n),
        mask=torch.ones(n),
        seq_length=torch.tensor(n),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("same_chain", [False, True])
def test_relative_encoding_in_the_target_dtype_equals_the_int64_encoding_cast(
    dtype, same_chain
):
    tokens = _tokens()
    old = featurization.create_relative_encoding(
        tokens, 32, 2, chain_bucket_on_same_chain=same_chain
    )
    new = featurization.create_relative_encoding(
        tokens, 32, 2, chain_bucket_on_same_chain=same_chain, dtype=dtype
    )
    assert old.dtype == torch.int64
    assert new.dtype == dtype
    assert torch.equal(new, old.to(dtype))


def _reembedding() -> ConfidenceReembedding:
    spec = SPECS["esmfold2"]
    torch.manual_seed(0)
    module = ConfidenceReembedding(
        c_single=16,
        c_pair=8,
        c_target_feat=spec.target_feat_channel,
        learned_bins=spec.confidence_learned_bins,
        esm_classes=spec.single_cond_layout == "esm",
        chain_bucket_on_same_chain=spec.chain_bucket_on_same_chain,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(0.3 * torch.randn_like(parameter))
    return module


def test_reembedding_split_equals_the_one_call_for_every_sample(monkeypatch):
    n = 12
    module = _reembedding()
    bonds = torch.randint(0, 7, (n, n))
    monkeypatch.setattr(head, "token_bond_types", lambda batch, symmetric: bonds)
    batch = SimpleNamespace(token_features=_tokens(n))
    generator = torch.Generator().manual_seed(1)
    pair = torch.randn(n, n, 8, generator=generator)
    single = torch.randn(n, 16, generator=generator)
    width = module.s_inputs_norm.normalized_shape[0]
    target = torch.randn(n, width - 4, generator=generator)
    pair_mask = torch.ones(n, n)
    samples = [4.0 * torch.randn(n, 3, generator=generator) for _ in range(3)]

    with torch.no_grad():
        terms = module.fold_terms(pair, single, target, batch)
        base = terms[0].clone()
        for positions in samples:
            expected = module(pair, single, target, positions, pair_mask, batch)
            got = module.sample_term(terms[0], positions, pair_mask), terms[1]
            assert torch.equal(got[0], expected[0])
            assert torch.equal(got[1], expected[1])
        assert torch.equal(terms[0], base)  # the shared pair is only read


def test_the_sample_loop_builds_the_fold_terms_once_per_fold(monkeypatch):
    seen = []

    def fake_forward(self, sample, *args: object) -> dict:
        seen.append(self._fold_cache)
        return {"x": sample.sum()[None]}

    monkeypatch.setattr(ConfidenceHead, "forward", fake_forward)
    spec = SPECS["esmfold2"]
    confidence = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(confidence)
    confidence.reembed_pair = spec.confidence_reembed_pair
    confidence.row_pool_attn = None
    confidence._fold_cache = None
    out = confidence._forward_samples(
        torch.ones(3, 2, 2, 3), {}, None, None, None, None
    )
    assert out["x"].shape == (3, 1)
    assert len(seen) == 3
    assert isinstance(seen[0], dict)
    assert all(item is seen[0] for item in seen)  # one dict for the fold's samples
    assert confidence._fold_cache is None


def test_fold_scope_opens_and_drops_the_cache():
    confidence = ConfidenceHead.__new__(ConfidenceHead)
    torch.nn.Module.__init__(confidence)
    confidence._fold_cache = None
    with confidence.fold_scope():
        assert confidence._fold_cache == {}
    assert confidence._fold_cache is None
    with pytest.raises(RuntimeError), confidence.fold_scope():
        raise RuntimeError
    assert confidence._fold_cache is None


def test_the_per_sample_confidence_calls_share_one_fold_scope():
    """All calls of a fold see the same cache dict.

    ``_confidence_over_samples`` calls the head once per sample (chunk 1).
    """
    from foldjet.models.architectures.af3 import AlphaFold3 as Model

    seen = []

    class Head(ConfidenceHead):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self._fold_cache = None

        def sample_chunk(self, count, tokens) -> int:
            return 1

        def forward(self, **kwargs: object) -> dict:
            seen.append(self._fold_cache)
            return {"x": kwargs["dense_atom_positions"].sum()[None]}

    batch = SimpleNamespace(
        token_features=SimpleNamespace(mask=None, asym_id=None),
        pseudo_beta_info=SimpleNamespace(token_atoms_to_pseudo_beta=None),
    )
    model = SimpleNamespace(confidence_head=Head(), _head_batch=lambda b: b)
    out = Model._confidence_over_samples(model, torch.ones(3, 4, 2, 3), {}, batch)
    assert out["x"].shape == (3, 1)
    assert len(seen) == 3
    assert isinstance(seen[0], dict)
    assert all(item is seen[0] for item in seen)
    assert model.confidence_head._fold_cache is None


def _stub_reembedding_head(monkeypatch) -> tuple[ConfidenceHead, dict[str, int]]:
    """A ``reembed_pair`` head with the heavy parts stubbed.

    No pairformer, a counting ``fold_terms``: the sample-independent state and its use.
    """
    calls = {"fold_terms": 0}

    class Reembedding(torch.nn.Module):
        def fold_terms(
            self, pair, single, target_feat, batch
        ) -> tuple[torch.Tensor, torch.Tensor]:
            calls["fold_terms"] += 1
            return pair * 2.0 + target_feat.sum(), single + 1.0

        def sample_term(self, pair, positions, pair_mask) -> torch.Tensor:
            return pair + positions.sum() * pair_mask[..., None]

    class Head(ConfidenceHead):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.spec = SimpleNamespace(
                confidence_global_norm=False, confidence_single_clamp=False
            )
            self.reembed_pair = True
            self.left_half_distance_logits = torch.nn.Linear(2, 2)
            self.reembedding = Reembedding()
            self.confidence_pairformer = []
            self.row_pool_attn = None
            self._fold_cache = None

        def _output_heads(self, pair, single, *args: object) -> dict:
            return {"pair": pair, "single": single}

    monkeypatch.setattr(
        head.atom_layout, "convert", lambda info, positions, layout_axes: positions
    )
    return Head(), calls


def _fold_inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(3)
    embeddings = {
        "pair": torch.randn(4, 4, 2, generator=generator),
        "single": torch.randn(4, 3, generator=generator),
        "target_feat": torch.randn(4, 5, generator=generator),
    }
    return (
        embeddings,
        torch.ones(4),
        [torch.randn(4, 3, generator=generator) for _ in range(3)],
    )


def test_a_call_with_the_fold_state_equals_the_call_that_builds_it(monkeypatch):
    confidence, calls = _stub_reembedding_head(monkeypatch)
    embeddings, seq_mask, samples = _fold_inputs()
    batch = SimpleNamespace()
    with torch.no_grad():
        expected = [
            confidence(s, embeddings, seq_mask, None, None, batch) for s in samples
        ]
        # no fold scope, no state: every call builds its own
        assert calls["fold_terms"] == len(samples)
        calls["fold_terms"] = 0
        state = confidence.fold_state(embeddings, seq_mask, batch)
        assert calls["fold_terms"] == 1
        base = [state["terms"][0].clone(), state["terms"][1].clone()]
        got = [
            confidence(s, embeddings, seq_mask, None, None, batch, fold=state)
            for s in samples
        ]
        # the calls that were given the state build nothing
        assert calls["fold_terms"] == 1
    for a, b in zip(expected, got, strict=True):
        assert torch.equal(a["pair"], b["pair"])
        assert torch.equal(a["single"], b["single"])
    # the state is only read
    assert torch.equal(state["terms"][0], base[0])
    assert torch.equal(state["terms"][1], base[1])


def test_the_sample_loop_builds_the_fold_state_once_and_hands_it_to_every_sample():
    from foldjet.models.architectures.af3 import AlphaFold3 as Model

    seen = {"built": 0, "folds": []}

    class Head(ConfidenceHead):
        def __init__(self, reembed) -> None:
            torch.nn.Module.__init__(self)
            self._fold_cache = None
            self.reembed_pair = reembed

        def sample_chunk(self, count, tokens) -> int:
            return 1

        def fold_state(self, embeddings, seq_mask, batch) -> dict:
            seen["built"] += 1
            return {"state": object()}

        def forward(self, **kwargs: object) -> dict:
            seen["folds"].append(kwargs.get("fold"))
            return {"x": kwargs["dense_atom_positions"].sum()[None]}

    batch = SimpleNamespace(
        token_features=SimpleNamespace(mask=None, asym_id=None),
        pseudo_beta_info=SimpleNamespace(token_atoms_to_pseudo_beta=None),
    )

    def run(reembed, samples) -> tuple[int, list]:
        seen["built"], seen["folds"] = 0, []
        model = SimpleNamespace(confidence_head=Head(reembed), _head_batch=lambda b: b)
        Model._confidence_over_samples(model, torch.ones(samples, 4, 2, 3), {}, batch)
        return seen["built"], list(seen["folds"])

    built, folds = run(reembed=True, samples=3)
    assert built == 1
    assert len(folds) == 3
    assert folds[0] is not None
    assert all(f is folds[0] for f in folds)
    # one sample: nothing to share
    assert run(reembed=True, samples=1) == (0, [None])
    # a head that does not re-embed has no fold state
    assert run(reembed=False, samples=3) == (0, [None] * 3)
