"""CPU tests of ``foldforge.train.step_graph``: the signature / key / static-buffer / cache logic with a FAKE capture (no CUDA graph is made here; the real
capture is exercised by the GPU gates in ``scripts/train_protenix1.py --bench --step-graph``)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from foldforge.train import step_graph
from foldforge.train.config import TrainConfig
from foldforge.train.data import TrainBatch
from foldforge.train.step import DeviceParts


@pytest.fixture(autouse=True)
def cpu_tensors_are_device(monkeypatch):
    """On the CPU every tensor is 'not CUDA': treat only ``*:input_shape`` as host metadata, as on a GPU."""
    monkeypatch.setattr(
        step_graph, "_is_host", lambda key, tensor: key.endswith(":input_shape")
    )


def make_example(tokens: int = 4, real: int = 3, fill: float = 1.0) -> TrainBatch:
    return TrainBatch(
        features={
            "aatype": torch.full((tokens,), 2, dtype=torch.int32),
            "msa": torch.full((6, tokens), 1, dtype=torch.int8),
            "token_atoms_to_pseudo_beta:input_shape": torch.tensor([real, 24]),
        },
        atom_positions=torch.full((tokens, 24, 3), fill),
        atom_mask=torch.ones(tokens, 24, dtype=torch.bool),
        present_index=torch.arange(8),
        present_valid=torch.ones(8, dtype=torch.bool),
    )


def test_signature_ignores_values_and_the_real_token_count():
    a, b = make_example(real=3, fill=1.0), make_example(real=4, fill=7.0)
    assert step_graph.example_signature(a) == step_graph.example_signature(b)


def test_signature_follows_shapes_dtypes_and_optional_fields():
    base = step_graph.example_signature(make_example())
    assert step_graph.example_signature(make_example(tokens=5)) != base
    wide = make_example()
    wide.features["msa"] = wide.features["msa"].to(torch.int16)
    assert step_graph.example_signature(wide) != base
    assert (
        step_graph.example_signature(
            replace(make_example(), bond_pairs=torch.zeros(2, 4, dtype=torch.long))
        )
        != base
    )
    assert step_graph.example_signature(replace(make_example(), atom_mask=None)) != base


def test_static_example_is_a_private_copy_and_load_overwrites_it():
    src = make_example(fill=1.0)
    static = step_graph.static_example(src)
    assert static.atom_positions.data_ptr() != src.atom_positions.data_ptr()
    assert static.features["msa"].data_ptr() != src.features["msa"].data_ptr()
    assert (
        static.features["token_atoms_to_pseudo_beta:input_shape"]
        is src.features["token_atoms_to_pseudo_beta:input_shape"]
    )  # host metadata shared
    new = make_example(fill=5.0)
    pointers = [t.data_ptr() for t in step_graph._device_leaves(static)]  # noqa: SLF001
    step_graph.load_example(static, new)
    assert [t.data_ptr() for t in step_graph._device_leaves(static)] == pointers  # noqa: SLF001 - in place: the graph reads these addresses
    assert torch.equal(static.atom_positions, new.atom_positions)
    assert src.atom_positions.eq(1.0).all()  # the source is untouched


def test_load_example_rejects_a_different_layout():
    static = step_graph.static_example(make_example())
    other = replace(make_example(), bond_pairs=torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(RuntimeError):
        step_graph.load_example(static, other)


def make_parts() -> DeviceParts:
    parts = DeviceParts(2, (4.0, 1.0, 0.03))
    parts.chunks = [
        (0.5, torch.tensor(1.0), torch.tensor(0.5), torch.tensor(3.0)),
        (0.5, torch.tensor(2.0), torch.tensor(0.25), torch.tensor(5.0)),
    ]
    parts.distogram = torch.tensor(0.7)
    parts.confidence = torch.tensor(6.0)
    parts.conf_parts = {"plddt": torch.tensor(1.0), "pde": torch.tensor(2.0)}
    return parts


def test_clone_parts_has_the_same_floats_and_independent_storage():
    parts = make_parts()
    cloned = step_graph.clone_parts(parts)
    assert cloned.floats() == parts.floats()
    parts.chunks[0][1].fill_(99.0)  # the static output of the next replay
    parts.distogram.fill_(99.0)
    assert cloned.floats() != parts.floats()
    assert cloned.chunks[0][1].item() == 1.0


def test_zero_grads_zeroes_in_place_and_keeps_none():
    a, b = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(2))
    a.grad = torch.full((3,), 4.0)
    pointer = a.grad.data_ptr()
    step_graph.zero_grads([a, b])
    assert a.grad.data_ptr() == pointer and a.grad.eq(0).all()
    assert b.grad is None


class FakeGraph:
    def __init__(self, entry_static, parts):
        self.static, self.parts, self.replays = entry_static, parts, 0

    def replay(self):
        self.replays += 1
        self.parts.distogram.fill_(
            self.static.atom_positions.sum()
        )  # the 'graph' reads the static input


def fake_capturer(graph: StepGraph | None = None):
    """A ``_capture`` that records the call and returns an entry over a fake graph."""
    calls = []

    def capture(self, model, example, cfg, passes, scale):  # noqa: ARG001
        calls.append((passes, scale))
        static = step_graph.static_example(example)
        parts = make_parts()
        return step_graph._Entry(FakeGraph(static, parts), static, parts, 0, [])  # noqa: SLF001

    return capture, calls


StepGraph = step_graph.StepGraph


def test_refuses_cpu_only_and_unsupported_rollout_placements():
    cfg = TrainConfig(step_graph=True)
    assert StepGraph.refuses(object(), make_example(), cfg) == "not a CUDA device"


def test_refuses_reasons_beyond_the_device():
    on_cuda = SimpleNamespace(
        atom_positions=SimpleNamespace(device=SimpleNamespace(type="cuda"))
    )
    model = object()
    assert "only" in StepGraph.refuses(
        model, on_cuda, TrainConfig(step_graph=True, only="trunk")
    )
    assert "side" in StepGraph.refuses(
        model, on_cuda, TrainConfig(step_graph=True, rollout="side")
    )
    assert (
        StepGraph.refuses(model, on_cuda, TrainConfig(step_graph=True, rollout="graph"))
        is None
    )
    assert (
        StepGraph.refuses(model, on_cuda, TrainConfig(step_graph=True, rollout="after"))
        is None
    )
    assert StepGraph.refuses(
        SimpleNamespace(spec=SimpleNamespace(structural_tokens=True)),
        on_cuda,
        TrainConfig(step_graph=True),
    )


def test_cache_one_capture_per_key_and_replay_loads_the_new_example(monkeypatch):
    monkeypatch.setattr(
        StepGraph, "refuses", staticmethod(lambda model, example, cfg: None)
    )
    capture, calls = fake_capturer()
    monkeypatch.setattr(StepGraph, "_capture", capture)
    cfg, model = TrainConfig(step_graph=True), object()
    sg = StepGraph(max_graphs=3)
    first = sg(model, make_example(fill=1.0), cfg, 2, 0.5)
    assert calls == [(2, 0.5)] and first.distogram.item() == pytest.approx(
        4 * 24 * 3 * 1.0
    )
    second = sg(
        model, make_example(fill=2.0, real=4), cfg, 2, 0.5
    )  # same key (other values / real token count): replay, no capture
    assert calls == [(2, 0.5)] and second.distogram.item() == pytest.approx(
        4 * 24 * 3 * 2.0
    )
    assert (
        second.distogram is not sg.entries[next(iter(sg.entries))].parts.distogram
    )  # a clone: the next replay cannot overwrite it
    sg(model, make_example(), cfg, 3, 0.5)  # another recycle count: another graph
    sg(model, make_example(), cfg, 2, 0.25)  # another loss scale (baked into the graph)
    assert calls == [(2, 0.5), (3, 0.5), (2, 0.25)]
    assert len(sg.entries) == 3
    assert sg.replays == 4


def test_cache_limit_runs_new_keys_eager(monkeypatch):
    monkeypatch.setattr(
        StepGraph, "refuses", staticmethod(lambda model, example, cfg: None)
    )
    capture, calls = fake_capturer()
    monkeypatch.setattr(StepGraph, "_capture", capture)
    cfg, model = TrainConfig(step_graph=True), object()
    sg = StepGraph(max_graphs=2)
    assert sg(model, make_example(tokens=4), cfg, 1) is not None
    assert sg(model, make_example(tokens=5), cfg, 1) is not None
    assert (
        sg(model, make_example(tokens=6), cfg, 1) is None
    )  # a third shape: the caller runs it eagerly
    assert (
        sg(model, make_example(tokens=4), cfg, 1) is not None
    )  # the captured ones keep replaying
    assert sg.fallbacks == 1 and len(calls) == 2


def test_failed_capture_marks_the_key_and_falls_back(monkeypatch):
    monkeypatch.setattr(
        StepGraph, "refuses", staticmethod(lambda model, example, cfg: None)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    attempts = []

    def broken(self, model, example, cfg, passes, scale):  # noqa: ARG001
        attempts.append(passes)
        raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(StepGraph, "_capture", broken)
    cfg, model = TrainConfig(step_graph=True), object()
    sg = StepGraph()
    assert sg(model, make_example(), cfg, 2) is None
    assert sg(model, make_example(), cfg, 2) is None  # not retried
    assert (
        attempts == [2]
        and sg.fallbacks == 2
        and "RuntimeError" in next(iter(sg.report()["failed"].values()))
    )
    # another key is still tried
    assert sg(model, make_example(), cfg, 3) is None and attempts == [2, 3]


def test_out_of_memory_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        StepGraph, "refuses", staticmethod(lambda model, example, cfg: None)
    )

    def oom(self, *args):  # noqa: ARG001
        raise torch.OutOfMemoryError("oom")

    monkeypatch.setattr(StepGraph, "_capture", oom)
    with pytest.raises(torch.OutOfMemoryError):
        StepGraph()(object(), make_example(), TrainConfig(step_graph=True), 1)


def test_mode_flag_scopes_and_restores():
    model = torch.nn.Linear(1, 1)
    assert step_graph.mode(model) is None and not step_graph.inline_rollout(model)
    with step_graph._mode(model, "capture"):  # noqa: SLF001
        assert step_graph.inline_rollout(model)
        with step_graph.bypass(model):
            assert step_graph.mode(model) == "bypass" and not step_graph.inline_rollout(
                model
            )
        assert step_graph.inline_rollout(model)
    assert step_graph.mode(model) is None


def test_step_graph_off_by_default():
    assert TrainConfig().step_graph is False
