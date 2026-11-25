"""Data parallelism without DDP + the speed-measurement helpers.

CPU: gloo with two spawned processes, fake sizes, fake observations.
"""

import multiprocessing
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from foldjet.train import bench, dist
from foldjet.train.config import TrainConfig


# ---- effective batch / world
def test_accum_for_effective_batch():
    assert dist.accum_for_effective_batch(256, 8) == 32
    assert dist.accum_for_effective_batch(256, 1) == 256
    assert dist.accum_for_effective_batch(256, 2) == 128
    with pytest.raises(ValueError, match="not divisible"):
        dist.accum_for_effective_batch(256, 3)
    with pytest.raises(ValueError, match="must be positive"):
        dist.accum_for_effective_batch(0, 1)


def test_world_info_from_environment():
    assert dist.world_info({}) == dist.WorldInfo(0, 1, 0)
    info = dist.world_info({"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "3"})
    assert (info.rank, info.world_size, info.local_rank, info.is_main) == (
        3,
        8,
        3,
        False,
    )
    assert dist.world_info({"RANK": "0", "WORLD_SIZE": "2"}).is_main


# ---- the micro-step plan
def test_global_micro_indices_partition_the_single_process_sequence():
    accum, world, steps = 3, 4, 5
    seen = []
    for it in range(steps):
        for rank in range(world):
            seen += [
                bench.global_micro_index(it, m, accum, rank, world)
                for m in range(accum)
            ]
    assert sorted(seen) == list(range(steps * accum * world))
    # rank r of W takes micro steps r, r + W, ... of the single-process run with
    # accum * W
    assert [bench.global_micro_index(0, m, accum, 1, world) for m in range(accum)] == [
        1,
        5,
        9,
    ]
    assert [bench.global_micro_index(2, m, 1, 0, 1) for m in range(1)] == [2]


def test_micro_plan_is_pure_and_seed_dependent():
    cfg = TrainConfig()
    a = bench.micro_plan(0, 17, 8, cfg)
    assert a == bench.micro_plan(0, 17, 8, cfg)
    assert 0 <= a.example < 8
    assert cfg.recycles_min <= a.passes <= cfg.recycles_max
    plans = {bench.micro_plan(0, i, 8, cfg) for i in range(200)}
    assert len({p.example for p in plans}) == 8
    assert len({p.passes for p in plans}) == 4
    assert bench.micro_plan(1, 17, 8, cfg) != a  # another base seed, another stream


def test_two_ranks_see_the_stream_of_the_single_process_run():
    cfg = TrainConfig()
    accum, world = 4, 2
    single = [
        bench.micro_plan(7, bench.global_micro_index(0, m, accum * world, 0, 1), 6, cfg)
        for m in range(accum * world)
    ]
    ranks = [
        [
            bench.micro_plan(7, bench.global_micro_index(0, m, accum, r, world), 6, cfg)
            for m in range(accum)
        ]
        for r in range(world)
    ]
    assert sorted(p.torch_seed for r in ranks for p in r) == sorted(
        p.torch_seed for p in single
    )
    assert list(single[0::2]) == ranks[0]
    assert list(single[1::2]) == ranks[1]


# ---- pool
def test_pick_pool_spreads_over_the_window(tmp_path: Path):
    sizes = {f"t{n:03d}-assembly1": n for n in range(100, 500, 5)}
    for name in sizes:
        (tmp_path / f"{name}.yaml").write_text("x")
    pool = bench.pick_pool(384, 8, sizes, tmp_path)
    tokens = [sizes[t] for t in pool]
    assert len(set(pool)) == 8
    assert tokens == sorted(tokens)
    assert min(tokens) >= int(0.7 * 384)
    assert max(tokens) <= 384
    assert max(tokens) >= 380
    with pytest.raises(ValueError, match="FoldBench targets have"):
        bench.pick_pool(384, 100, sizes, tmp_path)


def test_parse_pool_list():
    assert bench.parse_pool("8ayg-assembly1, 7fvu-assembly1") == [
        "8ayg-assembly1",
        "7fvu-assembly1",
    ]


# ---- engine table / summary
def test_engine_table_and_format():
    obs = {
        "modules": {
            "a.0": {
                "class": "TriangleMultiplication",
                "backend": "cueq",
                "served_host_calls": {"cueq": 3},
                "refused_host_calls": {},
            },
            "a.1": {
                "class": "TriangleMultiplication",
                "backend": "cueq",
                "served_host_calls": {"cueq": 1, "pytorch": 2},
                "refused_host_calls": {"grad": 2},
            },
            "b": {
                "class": "Transition",
                "backend": "pytorch",
                "served_host_calls": {"pytorch": 5},
                "refused_host_calls": {},
            },
        }
    }
    table = bench.engine_table(obs)
    assert table["TriangleMultiplication"] == {
        "modules": 2,
        "selected": {"cueq": 2},
        "served": {"cueq": 4, "pytorch": 2},
        "refused": {"grad": 2},
    }
    text = "\n".join(bench.format_table(table))
    assert "2x grad" in text
    assert "cueq:4,pytorch:2" in text
    assert "Transition" in text


def test_summarize_steps_reports_the_first_step_separately():
    s = bench.summarize_steps([10.0, 2.0, 4.0, 3.0])
    assert s["first_step_s"] == 10.0
    assert s["mean_s"] == 3.0
    assert s["median_s"] == 3.0
    assert s["min_s"] == 2.0
    assert s["max_s"] == 4.0
    assert s["total_wall_s"] == 19.0
    assert bench.summarize_steps([1.0])["mean_s"] is None


# ---- gradient all-reduce (gloo, 2 processes)
def _toy(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3))


def _micro_grads(model: nn.Module, index: int, scale: float) -> float:
    # the data of global micro step `index`
    gen = torch.Generator().manual_seed(1000 + index)
    x = torch.randn(4, 5, generator=gen)
    loss = model(x).pow(2).mean()
    (loss * scale).backward()
    return float(loss.detach())


def _worker(
    rank: int, world: int, init_file: str, accum: int, steps: int, out_dir: str
) -> None:
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    info = dist.WorldInfo(rank, world, rank)
    model = _toy(seed=rank)  # DIFFERENT init per rank: the broadcast must equalise it
    params = list(model.parameters())
    dist.broadcast_parameters(params, info)
    assert dist.ranks_agree(dist.param_checksum(params), info, torch.device("cpu"))
    losses = []
    for it in range(steps):
        for p in params:
            p.grad = None
        local = {"loss": 0.0}
        for m in range(accum):
            g = bench.global_micro_index(it, m, accum, rank, world)
            local["loss"] += _micro_grads(model, g, 1.0 / (accum * world)) / accum
        # tiny buckets: several all-reduces
        calls = dist.allreduce_gradients(params, info, bucket_numel=40)
        assert calls >= 2
        losses.append(dist.mean_over_ranks(local, info, torch.device("cpu"))["loss"])
        with torch.no_grad():
            for p in params:
                p.sub_(0.1 * p.grad)
    assert dist.ranks_agree(dist.param_checksum(params), info, torch.device("cpu"))
    if rank == 0:
        torch.save(
            {"losses": losses, "state": model.state_dict()}, Path(out_dir) / "dp.pt"
        )
    torch.distributed.destroy_process_group()


def test_allreduced_gradients_equal_the_single_process_accumulation():
    accum, world, steps = 3, 2, 3
    with tempfile.TemporaryDirectory() as tmp:
        init_file = str(Path(tmp) / "store")
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(r, world, init_file, accum, steps, tmp))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0
        dp = torch.load(Path(tmp) / "dp.pt")
    # single process: accum * world micro steps, scale 1 / (accum * world), the same
    # global micro indices, the same start weights (rank 0's init)
    model = _toy(seed=0)
    losses = []
    for it in range(steps):
        for p in model.parameters():
            p.grad = None
        total = 0.0
        for m in range(accum * world):
            total += _micro_grads(
                model,
                bench.global_micro_index(it, m, accum * world, 0, 1),
                1.0 / (accum * world),
            ) / (accum * world)
        losses.append(total)
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(0.1 * p.grad)
    assert dp["losses"] == pytest.approx(losses, rel=1e-6, abs=1e-7)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(dp["state"][key], value, rtol=1e-5, atol=1e-6)


def test_single_process_collectives_are_noops():
    info = dist.WorldInfo()
    model = _toy(0)
    params = list(model.parameters())
    before = [p.detach().clone() for p in params]
    dist.broadcast_parameters(params, info)
    assert dist.allreduce_gradients(params, info) == 0
    assert all(p.grad is None for p in params)
    assert dist.mean_over_ranks({"a": 1.5}, info, torch.device("cpu")) == {"a": 1.5}
    assert all(torch.equal(a, b) for a, b in zip(before, params, strict=True))
