"""The per-shape sampler graph cache (R26): key, eviction, stale fallback, no-API fallback. CPU only: the device side is replaced by fakes."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from foldforge.models import sampler_graph
from foldforge.models.sampler_graph import SamplerGraphCache, _Entry, _LayoutMismatch
from foldforge.modules.engine import capture as engine_capture


class Stale(RuntimeError):
    """Stands for the engine's ``StaleStaticInputs``."""


class FakeTables:
    def __init__(self) -> None:
        self.refreshed: list[object] = []
        self.released = 0
        self.fail = False

    def refresh(self, model=None) -> int:
        if self.fail:
            raise Stale("a weight changed")
        self.refreshed.append(model)
        return 3

    def release(self) -> int:
        self.released += 1
        return 1


class FakeCaptured:
    def __init__(self, tables, *, static_bytes=0, pool_bytes=0) -> None:
        self.tables = tables
        self.static_bytes = static_bytes
        self.pool_bytes = pool_bytes
        self.coords = torch.zeros(2, 3)
        self.calls: list[object] = []

    def __call__(self, pair, row=None) -> None:
        self.calls.append((pair, row))
        self.coords += 1


def entry_of(*, retained=0, leaves=None, tables=None, casts=(), masks=()) -> _Entry:
    tables = FakeTables() if tables is None else tables
    entry = _Entry(FakeCaptured(tables), None, leaves if leaves is not None else [torch.zeros(2)], list(casts), list(masks))
    entry.retained = retained
    return entry


# ---- bookkeeping ---------------------------------------------------------------------------------------------------------------------------------


def test_lookup_counts_hits_and_misses_and_orders_by_use() -> None:
    counters = SimpleNamespace(cache_hits=0, cache_misses=0, cache_evictions=0, uncached=0)
    cache = SamplerGraphCache(counters, max_entries=3)
    assert cache.lookup("a") is None
    cache.insert("a", entry_of(), 10**9)
    cache.insert("b", entry_of(), 10**9)
    assert cache.lookup("a") is not None  # a is now the most recently used
    assert list(cache.entries) == ["b", "a"]
    assert (counters.cache_hits, counters.cache_misses) == (1, 1)


def test_count_limit_evicts_least_recently_used_and_releases_its_tables() -> None:
    counters = SimpleNamespace(cache_evictions=0, uncached=0)
    cache = SamplerGraphCache(counters, max_entries=2)
    tables = {k: FakeTables() for k in "abc"}
    for key in "ab":
        cache.insert(key, entry_of(tables=tables[key]), 10**9)
    cache.lookup("a")
    cache.insert("c", entry_of(tables=tables["c"]), 10**9)
    assert list(cache.entries) == ["a", "c"]
    assert counters.cache_evictions == 1
    assert tables["b"].released == 1
    assert tables["a"].released == tables["c"].released == 0


def test_byte_budget_evicts_old_entries_and_an_oversized_entry_is_not_kept() -> None:
    counters = SimpleNamespace(cache_evictions=0, uncached=0)
    cache = SamplerGraphCache(counters, max_entries=8)
    assert cache.insert("a", entry_of(retained=40), 100)
    assert cache.insert("b", entry_of(retained=40), 100)
    assert cache.insert("c", entry_of(retained=40), 100)  # 120 > 100: a goes
    assert list(cache.entries) == ["b", "c"]
    assert counters.cache_evictions == 1
    big = entry_of(retained=101)
    assert not cache.insert("big", big, 100)
    assert "big" not in cache.entries
    assert counters.uncached == 1
    assert list(cache.entries) == ["b", "c"]  # nothing was evicted for it


def test_clear_releases_everything() -> None:
    cache = SamplerGraphCache(None, max_entries=4)
    tables = [FakeTables(), FakeTables()]
    for key, t in zip("ab", tables):
        cache.insert(key, entry_of(tables=t), 10**9)
    cache.clear()
    assert not cache.entries
    assert [t.released for t in tables] == [1, 1]


def test_released_entry_holds_nothing() -> None:
    entry = entry_of()
    entry.release()
    assert entry.captured is None and entry.leaves is None and entry.tables is None


def test_changed_weights_drop_every_graph() -> None:
    model = torch.nn.Linear(2, 2)
    cache = SamplerGraphCache(None, max_entries=4)
    cache.check_weights(model)
    cache.insert("a", entry_of(), 10**9)
    cache.check_weights(model)
    assert "a" in cache.entries  # same weights: kept
    with torch.no_grad():
        model.weight.add_(1.0)  # in place: version changes
    cache.check_weights(model)
    assert not cache.entries
    cache.insert("a", entry_of(), 10**9)
    model.weight = torch.nn.Parameter(model.weight.detach().clone())  # replaced: new address
    cache.check_weights(model)
    assert not cache.entries


def test_release_all_caches_reaches_every_live_cache() -> None:
    caches = [SamplerGraphCache(None) for _ in range(2)]
    for cache in caches:
        cache.insert("a", entry_of(), 10**9)
    sampler_graph.release_all_caches()
    assert all(not cache.entries for cache in caches)


# ---- an entry serves another fold -----------------------------------------------------------------------------------------------------------------


def test_entry_load_writes_in_place_refreshes_and_rewrites_the_casts() -> None:
    static = [torch.zeros(2, 2), torch.zeros(3)]
    pointers = [t.data_ptr() for t in static]
    pair = static[0]
    cast = torch.zeros(1, 2, 2, dtype=torch.bfloat16)
    flat_mask = torch.zeros(3)
    atom_mask = static[1]
    entry = entry_of(leaves=static, casts=[(pair, cast)], masks=[(flat_mask, atom_mask)])
    model = object()
    new = [torch.full((2, 2), 3.0), torch.tensor([1.0, 0.0, 1.0])]
    assert entry.load(new, model) == 3
    assert [t.data_ptr() for t in static] == pointers  # same storage: the graph and the engine's records keep reading it
    assert torch.equal(static[0], new[0]) and torch.equal(static[1], new[1])
    assert torch.equal(cast[0].float(), new[0])  # the cast copy follows its source
    assert torch.equal(flat_mask, new[1])
    assert entry.tables.refreshed == [None]  # the handle selects this graph's tables: no root filter


def test_entry_load_rejects_a_fold_of_another_layout() -> None:
    entry = entry_of(leaves=[torch.zeros(2)])
    with pytest.raises(_LayoutMismatch):
        entry.load([torch.zeros(2), torch.zeros(2)], None)


# ---- the sampler call through the cache (device side faked) ---------------------------------------------------------------------------------------


class CpuCache(SamplerGraphCache):
    """The cache with the device queries replaced."""

    budget_bytes = 10**9

    def build(self, make):
        return make()


@pytest.fixture
def cached_call(monkeypatch):
    """``run(fold_data, ...)``: ``sample`` with a fake fold, engine signature and capture."""
    monkeypatch.setattr(engine_capture, "signature", lambda root, *tensors: (id(root), tuple(None if t is None else tuple(t.shape) for t in tensors)))
    monkeypatch.setattr(engine_capture, "stale_error", lambda: Stale)
    monkeypatch.setattr(sampler_graph, "_flatten", lambda tree: (tuple(tuple(t.shape) for t in tree), list(tree)))
    captures: list[_Entry] = []

    def fake_capture(cache, fold, sampler, tree, coords, sigmas, row, **_):
        leaves = [t.clone() for t in tree]
        entry = entry_of(leaves=leaves)
        captures.append(entry)
        return entry

    monkeypatch.setattr(sampler_graph, "_capture_entry", fake_capture)
    counters = SimpleNamespace(cache_hits=0, cache_misses=0, cache_evictions=0, uncached=0, refreshes=0, stale_recaptures=0)
    cache = CpuCache(counters, max_entries=4)
    model = torch.nn.Linear(1, 1)
    sampler = SimpleNamespace(config=SimpleNamespace(gamma_0=0.8))

    def run(tree, *, steps=3, generator=None):
        fold = SimpleNamespace(model=model, tree=lambda: tree, casts=lambda: [])
        coords = torch.zeros(2, 3)
        table = torch.arange(float(steps)).reshape(steps, 1)
        pairs = torch.zeros(steps, 2)
        sigmas = torch.zeros(steps + 1)
        return sampler_graph._sample_cached(
            cache, fold, sampler, coords, table, pairs, sigmas, generator=generator, coordinate_dims=2, scopes=counters, use_glue=False
        )

    return SimpleNamespace(run=run, cache=cache, counters=counters, captures=captures)


def test_first_fold_captures_and_later_folds_of_the_shape_hit(cached_call) -> None:
    a, b = [torch.ones(2, 2), torch.ones(3)], [torch.full((2, 2), 2.0), torch.full((3,), 2.0)]
    cached_call.run(a)
    cached_call.run(b)
    cached_call.run(a)
    assert len(cached_call.captures) == 1
    assert (cached_call.counters.cache_misses, cached_call.counters.cache_hits, cached_call.counters.refreshes) == (1, 2, 2)
    static = cached_call.captures[0].leaves
    assert torch.equal(static[0], a[0])  # the last fold written into the static buffers
    assert len(cached_call.captures[0].captured.calls) == 9  # 3 steps per fold, one graph


def test_another_shape_is_another_graph_and_steps_do_not_matter(cached_call) -> None:
    cached_call.run([torch.ones(2, 2)], steps=3)
    cached_call.run([torch.ones(2, 2)], steps=5)  # same shape, other step count: the same graph
    cached_call.run([torch.ones(4, 2)], steps=3)
    assert len(cached_call.captures) == 2
    assert cached_call.counters.cache_hits == 1


def test_a_stale_refresh_recaptures_once_and_counts_it(cached_call) -> None:
    tree = [torch.ones(2, 2)]
    cached_call.run(tree)
    cached_call.captures[0].tables.fail = True
    result = cached_call.run(tree)
    assert result is not None
    assert cached_call.counters.stale_recaptures == 1
    assert len(cached_call.captures) == 2
    assert cached_call.captures[0].tables is None  # released
    cached_call.run(tree)  # the fresh graph serves the next fold
    assert len(cached_call.captures) == 2


def test_an_entry_without_an_engine_handle_is_used_once_and_not_kept(cached_call, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    def no_handle(*args, **kwargs):
        entry = entry_of(leaves=[torch.zeros(2, 2)])
        entry.tables = None
        return entry

    monkeypatch.setattr(sampler_graph, "_capture_entry", no_handle)
    cached_call.run([torch.ones(2, 2)])
    assert not cached_call.cache.entries
    assert cached_call.counters.uncached == 1


def test_a_tree_the_cache_cannot_describe_falls_back(cached_call, monkeypatch) -> None:
    def unsupported(tree):
        raise TypeError("Unsupported graph input")

    monkeypatch.setattr(sampler_graph, "_flatten", unsupported)
    assert cached_call.run([torch.ones(2)]) is None  # the caller runs the per-fold capture
    assert cached_call.counters.uncached == 1


def test_changed_weights_recapture(cached_call) -> None:
    tree = [torch.ones(2, 2)]
    cached_call.run(tree)
    model = cached_call.cache.weights  # token of the model the fold carries
    assert model is not None
    cached_call.cache.weights = (0, 0)  # the weights the graph was captured with are not the model's now
    cached_call.run(tree)
    assert len(cached_call.captures) == 2


# ---- when the cache is not used --------------------------------------------------------------------------------------------------------------------


def test_training_and_missing_fold_use_no_cache() -> None:
    counters = SimpleNamespace(uncached=0)
    scope = SimpleNamespace(cache=SamplerGraphCache(counters), uncached=0)
    scope.cache.counters = scope
    fold = object()
    assert sampler_graph._cache_of(scope, None) is None
    with torch.enable_grad():
        assert sampler_graph._cache_of(scope, fold) is None
    assert scope.uncached == 1  # the training fold is counted
    with torch.no_grad():
        assert sampler_graph._cache_of(scope, fold) is scope.cache
    assert sampler_graph._cache_of(SimpleNamespace(), fold) is None  # a scope without a cache (config off, engine without the API)


def _fake_engine_capture(monkeypatch, **attributes):
    module = types.ModuleType("miniworld_engine.capture")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "miniworld_engine.capture", module)
    parent = sys.modules.get("miniworld_engine")
    if parent is not None:
        monkeypatch.setattr(parent, "capture", module, raising=False)


def test_an_engine_without_the_refresh_api_gets_no_cache(monkeypatch) -> None:
    from foldforge.models.execution import SamplerScope

    # the 2.5.4 module: static_inputs is there, the refresh API is not
    _fake_engine_capture(monkeypatch, static_inputs=lambda: None, static_weights=lambda: None, clear_weight_caches=lambda: None)
    assert not engine_capture.refresh_available()
    assert engine_capture.stale_error().__name__ == "_NeverRaised"
    config = SimpleNamespace(sampler_graph_cache=True, max_graphs=4, sampler_graph_cache_fraction=0.3)
    scope = SamplerScope(SimpleNamespace(bypass=0), config)
    assert scope.cache is None
    assert sampler_graph._cache_of(scope, object()) is None  # every fold is captured on its own


def test_an_engine_with_the_refresh_api_gets_one_and_the_config_can_turn_it_off(monkeypatch) -> None:
    from foldforge.models.execution import SamplerScope

    api = dict.fromkeys(("refresh_static_inputs", "StaticTables", "signature", "StaleStaticInputs"), object)
    _fake_engine_capture(monkeypatch, **api)
    assert engine_capture.refresh_available()
    on = SimpleNamespace(sampler_graph_cache=True, max_graphs=3, sampler_graph_cache_fraction=0.25)
    scope = SamplerScope(SimpleNamespace(bypass=0), on)
    assert scope.cache is not None and scope.cache.max_entries == 3 and scope.cache.budget_fraction == 0.25
    off = SimpleNamespace(sampler_graph_cache=False, max_graphs=3, sampler_graph_cache_fraction=0.25)
    assert SamplerScope(SimpleNamespace(bypass=0), off).cache is None


def test_scope_counters_reset_and_report_names() -> None:
    from foldforge.models.execution import SamplerScope

    scope = SamplerScope(SimpleNamespace(bypass=0))
    scope.captures = scope.refreshes = scope.cache_hits = 5
    scope.reset_counters()
    assert all(getattr(scope, name) == 0 for name in SamplerScope.COUNTERS)
    assert {"captures", "replays", "refreshes", "cache_hits", "cache_misses", "cache_evictions", "stale_recaptures", "uncached"} <= set(SamplerScope.COUNTERS)


# ---- the per-fold state of the model --------------------------------------------------------------------------------------------------------------


def test_grouped_fold_tree_drops_the_epochs_and_bind_restores_the_state() -> None:
    from foldforge.models.grouped import GroupedSampler

    sampler = object.__new__(GroupedSampler)
    sampler.state = {"L": 4, "pair_all": torch.zeros(1), "cond": [{"epoch": 7, "base": torch.zeros(1)}, {"epoch": 8, "base": torch.ones(1)}]}
    tree = sampler.fold_tree()
    assert "epoch" not in tree["cond"][0] and tree["L"] == 4 and len(tree["cond"]) == 2
    original = dict(sampler.state)
    static = {"L": 4, "pair_all": torch.ones(1), "cond": [{"base": torch.ones(1)}]}
    with sampler.bind_fold(static):
        assert sampler.state["pair_all"] is static["pair_all"]
    assert sampler.state == original and sampler.state["cond"] is original["cond"]


def test_head_adopts_a_fold_under_the_key_of_its_batch_and_embeddings() -> None:
    from foldforge.modules.dense.diffusion_head import DiffusionHead

    head = object.__new__(DiffusionHead)
    torch.nn.Module.__init__(head)
    head._fold, head._fold_epoch = None, 3
    batch, embeddings = object(), {"single": torch.zeros(2), "pair": torch.zeros(2, 2)}
    tree = {"single_base": torch.zeros(1), "pair_cond": torch.zeros(2), "lite": {"single": embeddings["single"]}, "key_mask": torch.ones(1), "atom_state": None}
    head.adopt_fold(batch, embeddings, tree)
    entry = head._fold
    assert entry[0] == head._fold_key(batch, embeddings, True)  # what _conditioned looks up
    assert head._conditioned(batch, embeddings, True)[1] is tree["pair_cond"]  # served, not remade
    assert head.fold_tree().keys() == tree.keys() and head.fold_tree()["pair_cond"] is tree["pair_cond"]
    embeddings["single"].add_(1)  # a changed trunk output is another fold
    assert entry[0] != head._fold_key(batch, embeddings, True)


def test_token_dit_stack_reports_the_pair_cast_it_made() -> None:
    from foldforge.modules.engine.transformers import TokenDiTStack

    stack = TokenDiTStack([])
    assert stack.engine_pair_cast() is None
    pair = torch.zeros(2, 2)
    stack._pair_source, stack._pair_engine = (pair, pair._version), pair[None]  # a view: no copy to rewrite
    assert stack.engine_pair_cast() is None
    copy = pair.to(torch.bfloat16)[None]
    stack._pair_engine = copy
    found = stack.engine_pair_cast()
    assert found is not None and found[0] is pair and found[1] is copy


# ---- what the graph reads from the engine's caches -----------------------------------------------------------------------------------------------


def test_hold_new_keeps_the_engine_entries_made_since_the_marks(monkeypatch) -> None:
    stores = {"packs": {"old": torch.zeros(2)}, "chunks": {"k": torch.zeros(3)}}
    _fake_engine_capture(monkeypatch, weight_caches=lambda: stores)
    marks = engine_capture.cache_marks()
    stores["packs"]["new"] = torch.ones(4)  # made during the warm-up / capture
    stores["chunks"]["k"] = torch.ones(5)  # replaced by another capture's entry
    held = engine_capture.hold_new(marks)
    keys = sorted(str(key) for entries in held for key, _ in entries)
    assert keys == ["k", "new"]  # the untouched old entry is not ours to hold
    pack_entry = {key: value for entries in held for key, value in entries}
    stores["chunks"].clear()  # the engine evicts: the graph's references stay
    assert pack_entry["k"].numel() == 5
    assert engine_capture.tensor_bytes(held) == 0  # CPU tensors are not counted (CUDA bytes only)


def test_hold_helpers_do_nothing_on_an_engine_without_weight_caches(monkeypatch) -> None:
    _fake_engine_capture(monkeypatch)
    assert engine_capture.cache_marks() == {}
    assert engine_capture.hold_new({}) == []


def test_static_clone_keeps_expanded_dims_expanded_and_load_writes_the_unique_part() -> None:
    shared = torch.arange(6.0).reshape(1, 6).expand(4, 6)  # one mask served to 4 folds
    assert shared.stride(0) == 0
    static = sampler_graph._static_clone(shared)
    assert static.stride(0) == 0 and torch.equal(static, shared)
    assert sampler_graph._static_clone(torch.ones(3, 2)).stride() == (2, 1)
    pointer = static.data_ptr()
    entry = entry_of(leaves=[static, torch.zeros(2)])
    new = torch.full((1, 6), 7.0).expand(4, 6)
    entry.load([new, torch.ones(2)], None)  # a plain copy_ into an expanded tensor would raise
    assert static.data_ptr() == pointer and static.stride(0) == 0 and torch.equal(static, new)


def test_a_fold_larger_than_the_cutoff_is_not_cached(cached_call, monkeypatch) -> None:
    monkeypatch.setattr(engine_capture, "tensor_bytes", lambda held, seen=None: 3 << 30)
    assert cached_call.run([torch.ones(2, 2)]) is None  # the caller runs the per-fold capture: no static copy next to the fold
    assert cached_call.counters.uncached == 1 and not cached_call.captures
    assert cached_call.cache.max_fold_bytes == 2 << 30
