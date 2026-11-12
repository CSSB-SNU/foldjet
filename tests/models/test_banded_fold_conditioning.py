"""Banded pair conditioning (`DiffusionHead._fold_conditioning_banded`) equals the unbanded path bit for bit, and per-fold release (b)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from foldforge.models import execution
from foldforge.models.architectures.af3 import AlphaFold3
from foldforge.models.execution import ExecutedCallable, Execution
from foldforge.modules.dense import diffusion_head as dh
from foldforge.modules.dense import spec as specs
from foldforge.modules.dense.diffusion_head import DiffusionHead
from foldforge.modules.engine.transformers import TokenDiTStack

LENGTH = 21  # not a multiple of the band rows below


def _tokens(length):
    index = torch.arange(length)
    asym = (index >= length // 2).long()
    return SimpleNamespace(
        residue_index=index % 7, token_index=index, asym_id=asym, entity_id=(index >= length - 3).long(), sym_id=asym,
    )


def _inputs(head, spec, length, generator):
    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    target = head.c_single_cond_initial - spec.seq_channel - 2 * head.padded_single_cond
    embeddings = {"single": rand(length, spec.seq_channel), "pair": rand(length, length, spec.pair_channel), "target_feat": rand(length, target)}
    if head.pair_init_cond:
        embeddings["pair_init"] = rand(length, length, spec.pair_channel)
    return SimpleNamespace(token_features=_tokens(length)), embeddings


@pytest.fixture
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("name", ["ALPHAFOLD3", "BOLTZ2", "CHAI1", "OPENDDE"])
@pytest.mark.parametrize("use_conditioning", [True, False])
def test_banded_conditioning_equals_unbanded(monkeypatch, one_thread, name, use_conditioning):
    # A multi-threaded CPU GEMM partitions its rows by M (a band has fewer rows than the whole pair), which changes the last bit; one thread
    # runs the same per-row arithmetic. The H100 equality is checked on the real model (5I28 OpenDDE, n=8 / 12).
    spec = getattr(specs, name)
    torch.manual_seed(0)
    head = DiffusionHead(spec).eval()
    if head.pair_only_trunk or spec.single_cond_layout == "esm":
        pytest.skip("not a layout of this test")
    batch, embeddings = _inputs(head, spec, LENGTH, torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = head._fold_conditioning(batch, embeddings, use_conditioning)
        assert head._pair_band_rows(embeddings["pair"]) is None  # the default threshold: the unbanded code ran
        row_bytes = LENGTH * spec.pair_channel * 4
        monkeypatch.setattr(dh, "PAIR_BAND_THRESHOLD_BYTES", 0)
        monkeypatch.setattr(dh, "PAIR_BAND_BYTES", 4 * row_bytes + 1)  # 4 rows per band: 6 bands, the last one short
        assert head._pair_band_rows(embeddings["pair"]) == 4
        got = head._fold_conditioning(batch, embeddings, use_conditioning)
    assert all(torch.equal(a, b) for a, b in zip(want, got, strict=True))
    assert got[1].shape == want[1].shape


def test_threshold_is_in_bytes_and_default_is_a_no_op_for_small_pairs():
    head = DiffusionHead(specs.ALPHAFOLD3)
    small = torch.empty(64, 64, specs.ALPHAFOLD3.pair_channel)
    assert head._pair_band_rows(small) is None
    big = torch.empty(1, 1, 1).expand(4096, 4096, 128)  # a 4096^2 x 128 pair: 8 GiB as fp32
    assert head._pair_band_rows(big) == max(1, dh.PAIR_BAND_BYTES // (4096 * 128 * 4))


def test_relative_encoding_rows_are_rows_of_the_whole():
    from foldforge.modules.dense import featurization

    tokens = _tokens(LENGTH)
    whole = featurization.create_relative_encoding(tokens, 32, 2, dtype=torch.float32)
    band = featurization.create_relative_encoding(tokens, 32, 2, dtype=torch.float32, rows=slice(5, 9))
    assert torch.equal(band, whole[5:9])


def _graphed(epoch_key=None):
    config = SimpleNamespace(compile=False, cuda_graph=True, max_graphs=4)
    held = ExecutedCallable(lambda x: x, config, "test.site", epoch_key=epoch_key)
    held.graphs["signature"] = object()
    held.epoch = 3
    return held


def test_release_fold_drops_only_per_fold_graphs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    per_fold, cached = _graphed("fold_epoch"), _graphed()
    assert per_fold.release_fold() is True
    assert not per_fold.graphs
    assert per_fold.epoch is None
    assert cached.release_fold() is False
    assert len(cached.graphs) == 1  # a signature-cached trunk graph survives the next fold
    assert per_fold.release_fold() is False  # nothing left


def test_release_fold_state_is_skipped_for_a_small_previous_fold(monkeypatch):
    """A small kept state is not released early (gc + synchronize + rebuilt per-fold graphs cost 8-50% of a small warm forward)."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    per_fold = _graphed("fold_epoch")
    head = DiffusionHead.__new__(DiffusionHead)
    nn.Module.__init__(head)
    head._fold = (torch.zeros(4),)
    head.transformer = TokenDiTStack([])
    head.transformer._pair_source, head.transformer._pair_engine = (torch.zeros(8), 0), torch.zeros(8)
    assert head.fold_state_bytes() == 16 + 32 + 32  # the fold conditioning, the kept pair and its engine-dtype cast
    run = Execution.__new__(Execution)
    run.model, run.wrappers = SimpleNamespace(diffusion_head=head), [per_fold]
    run.release_fold_state()
    assert head._fold is not None
    assert per_fold.graphs  # untouched below the threshold
    monkeypatch.setattr(execution, "RELEASE_MIN_BYTES", 80)  # at or above it the state goes
    monkeypatch.setattr(execution.importlib, "import_module", lambda name: SimpleNamespace())
    run.release_fold_state()
    assert head._fold is None
    assert not per_fold.graphs


def test_release_fold_state_of_the_execution(monkeypatch):
    monkeypatch.setattr(execution, "RELEASE_MIN_BYTES", 0)  # the release itself (the gate has its own test)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    per_fold, cached = _graphed("fold_epoch"), _graphed()
    head = DiffusionHead.__new__(DiffusionHead)  # the real method over a bare instance: only the attributes it touches
    nn.Module.__init__(head)
    head._fold = ("previous fold",)
    head.transformer = TokenDiTStack([])
    head.transformer._pair_source, head.transformer._pair_engine = (torch.zeros(1), 0), torch.zeros(1)
    run = Execution.__new__(Execution)
    run.model, run.wrappers = SimpleNamespace(diffusion_head=head), [per_fold, cached]
    calls = []
    monkeypatch.setattr(execution.importlib, "import_module", lambda name: SimpleNamespace(release_dead_inputs=lambda: calls.append(name)))
    run.release_fold_state()
    assert head._fold is None
    assert head.transformer._pair_source is None
    assert head.transformer._pair_engine is None  # the kept pair (the previous fold's conditioning) is released
    assert not per_fold.graphs
    assert len(cached.graphs) == 1
    assert calls == ["miniworld_engine.kernels._capture"]


def test_release_fold_state_tolerates_an_engine_without_release_dead_inputs(monkeypatch):
    run = Execution.__new__(Execution)
    run.model, run.wrappers = SimpleNamespace(), []
    monkeypatch.setattr(execution.importlib, "import_module", lambda name: SimpleNamespace())
    run.release_fold_state()


def test_run_trunk_releases_the_previous_fold_first():
    order = []

    class Stop(Exception):
        pass

    def prepare(batch):
        order.append("trunk")
        raise Stop

    fake = SimpleNamespace(foldforge_release_fold_state=lambda: order.append("release"), _prepare_batch=prepare)
    with pytest.raises(Stop):
        AlphaFold3.run_trunk(fake, {"x": torch.zeros(1)})
    assert order == ["release", "trunk"]
