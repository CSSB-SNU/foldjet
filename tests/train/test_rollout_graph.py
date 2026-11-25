"""The training rollout graph (``train/rollout_graph.py``).

Shape key, cache budget / eviction, fallbacks, step hook, inline sampler. CPU only
(the device side is faked).
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from foldjet.diffusion.edm.sampling import EulerSampler
from foldjet.models import sampler_graph
from foldjet.train import rollout_graph, step
from foldjet.train.config import TrainConfig
from foldjet.train.rollout_graph import RolloutGraph, _Entry, graph_key, graph_tree


@dataclasses.dataclass
class Tokens:
    seq_length: torch.Tensor | None
    mask: torch.Tensor


@dataclasses.dataclass
class Batch:
    msa: torch.Tensor | None
    templates: torch.Tensor | None
    mask: torch.Tensor
    token_features: Tokens | None = None


def fake_entry(retained: int = 0, leaves=None) -> _Entry:
    entry = _Entry(
        SimpleNamespace(replay=lambda: None),
        None,
        leaves if leaves is not None else [torch.zeros(2)],
        torch.zeros(1, 3),
        0,
        [],
    )
    entry.retained = retained
    return entry


def test_tree_drops_msa_and_templates_and_keeps_the_trunk_outputs() -> None:
    batch = Batch(
        msa=torch.zeros(4, 4),
        templates=torch.zeros(2),
        mask=torch.ones(3),
        token_features=Tokens(seq_length=torch.tensor([268]), mask=torch.ones(3)),
    )
    tree = graph_tree(
        batch, {"pair": torch.zeros(3, 3, 2), "single": torch.zeros(3, 4)}
    )
    assert tree["batch"].msa is None
    assert tree["batch"].templates is None
    assert tree["batch"].mask is batch.mask
    # the crop length is not part of the key
    assert tree["batch"].token_features.seq_length is None
    assert batch.token_features.seq_length is not None
    assert set(tree["emb"]) == {"pair", "single"}


def test_key_is_the_layout_steps_and_model_not_the_values() -> None:
    model = object()
    a, b = ("layout-a",), ("layout-b",)
    assert graph_key(model, a, 20) == graph_key(model, a, 20)
    assert graph_key(model, a, 20) != graph_key(model, b, 20)
    assert graph_key(model, a, 20) != graph_key(model, a, 10)
    assert graph_key(model, a, 20) != graph_key(object(), a, 20)


def test_cache_evicts_least_recently_used_over_the_graph_count_and_byte_budget() -> (
    None
):
    graph = RolloutGraph(max_graphs=2)
    cache = graph.cache
    cache.insert("a", fake_entry(), 10**9)
    cache.insert("b", fake_entry(), 10**9)
    assert cache.lookup("a") is not None  # a is now the most recently used
    cache.insert("c", fake_entry(), 10**9)
    assert list(cache.entries) == ["a", "c"]
    assert graph.cache_evictions == 1
    assert graph.cache_hits == 1
    # bytes: an entry over the whole budget is not kept; two over it evict the older one
    assert cache.insert("huge", fake_entry(retained=100), 50) is False
    assert graph.uncached == 1
    cache.insert("d", fake_entry(retained=30), 50)
    cache.insert("e", fake_entry(retained=30), 50)
    assert list(cache.entries) == ["e"]
    report = graph.report()
    assert report["entries"] == 1
    assert report["evictions"] >= 2
    assert report["disabled"] is None


def test_entry_load_writes_the_static_buffers_and_checks_the_layout() -> None:
    static = [torch.zeros(2, 2), torch.zeros(3)]
    entry = fake_entry(leaves=static)
    entry.load([torch.ones(2, 2), torch.full((3,), 2.0)])
    assert torch.equal(static[0], torch.ones(2, 2))
    assert torch.equal(static[1], torch.full((3,), 2.0))
    with pytest.raises(RuntimeError):
        entry.load([torch.ones(2, 2)])
    # a stride-0 leaf: written through its unique part
    expanded = torch.zeros(1, 4).expand(5, 4)
    fake_entry(leaves=[expanded]).load([torch.ones(1, 4).expand(5, 4)])
    assert torch.equal(expanded, torch.ones(5, 4))


def test_cpu_inputs_fall_back_to_the_eager_rollout() -> None:
    graph = RolloutGraph()
    batch = Batch(
        msa=None,
        templates=None,
        mask=torch.ones(3),
        token_features=Tokens(seq_length=None, mask=torch.ones(3)),
    )
    model = SimpleNamespace(spec=SimpleNamespace(structural_tokens=False))
    assert graph(model, {}, batch, {"pair": torch.zeros(3, 3, 2)}, 20) is None
    assert graph.fallbacks == 1
    calls = []
    model.__dict__.pop("_foldjet_rollout_graph", None)
    out = rollout_graph.mini_rollout(
        model,
        {},
        batch,
        {"pair": torch.zeros(3, 3, 2)},
        20,
        lambda *a: calls.append(a) or "eager",
    )
    assert out == "eager"
    assert len(calls) == 1
    # the graph is made once per model
    assert model.__dict__["_foldjet_rollout_graph"].fallbacks == 1


def test_structural_tokens_are_not_served() -> None:
    model = SimpleNamespace(spec=SimpleNamespace(structural_tokens=True))
    assert "structural" in RolloutGraph.serves(model, torch.device("cuda"))
    assert RolloutGraph.serves(SimpleNamespace(spec=None), torch.device("cuda")) is None
    assert RolloutGraph.serves(model, torch.device("cpu")) == "not a CUDA device"


def test_step_hook_routes_to_the_graph_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = []
    monkeypatch.setattr(
        rollout_graph, "mini_rollout", lambda *a: seen.append(a) or "graph-result"
    )
    assert step.mini_rollout("m", {}, "b", {}, 20, graph=True) == "graph-result"
    assert len(seen) == 1
    # the eager rollout is the fallback handed over
    assert seen[0][-1] is step.mini_rollout


def test_rollout_config_accepts_graph_and_defaults_to_after() -> None:
    assert TrainConfig().rollout == "after"
    assert dataclasses.replace(TrainConfig(), rollout="graph").rollout == "graph"


def test_inline_sampler_equals_the_eager_sampler_for_the_same_noise() -> None:
    """``sample_inline`` has the arithmetic of ``EulerSampler.sample``.

    Device-scalar churn branch, no host read: same noise, same trajectory.
    """
    sampler = EulerSampler(EulerSampler.Config())
    sigmas = torch.tensor([160.0, 40.0, 8.0, 1.5, 0.3, 0.05])
    weight = torch.tensor([[0.9, 0.1, 0.0], [0.0, 1.1, 0.2], [0.1, 0.0, 0.8]])

    def denoise(coords, sigma) -> torch.Tensor:
        return (
            coords @ weight / (1 + sigma.reshape(-1, *(1,) * (coords.ndim - 1)) ** 0.5)
        )

    shape = (1, 5, 3)
    torch.manual_seed(0)
    eager = sampler.sample(
        denoise,
        shape,
        sigmas,
        device="cpu",
        chunk_size=None,
        initialize_all=True,
        coordinate_dims=1,
    )
    torch.manual_seed(0)
    inline = sampler_graph.sample_inline(
        sampler,
        denoise,
        shape,
        sigmas,
        device=torch.device("cpu"),
        generator=None,
        augment=None,
        align=None,
        coordinate_dims=1,
    )
    torch.testing.assert_close(inline, eager, rtol=1e-5, atol=1e-5)


def test_bump_versions_touches_every_parameter_and_buffer() -> None:
    model = torch.nn.BatchNorm1d(3)
    before = [t._version for t in (*model.parameters(), *model.buffers())]
    rollout_graph.bump_versions(model)
    after = [t._version for t in (*model.parameters(), *model.buffers())]
    assert all(b > a for a, b in zip(before, after, strict=True))


def test_autocast_cache_guard_is_a_noop_without_cuda_autocast() -> None:
    with rollout_graph._no_autocast_cache():
        assert not torch.is_autocast_enabled("cuda")
