"""``foldforge.train.fused_opt`` against the per-tensor path of ``train/optim.py``.

CPU, tiny model: clip, Adam, EMA, lr schedule, frozen / None-grad parameters, groups.
"""

import copy
from typing import Any

import pytest
import torch
from torch import nn

from foldforge.train import fused_opt, optim
from foldforge.train.config import TrainConfig


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evoformer = nn.Linear(5, 7)
        self.frozen = nn.Linear(3, 3)
        self.diffusion_head = nn.Linear(7, 4)
        self.confidence_head = nn.Linear(4, 2)
        self.unused = nn.Linear(2, 2)  # trains but never gets a gradient (grad None)
        for p in self.frozen.parameters():
            p.requires_grad_(requires_grad=False)


def _cfg(**kw: Any) -> TrainConfig:
    base = {
        "max_lr": 1e-2,
        "min_lr": 1e-3,
        "warmup_steps": 3,
        "decay_steps": 2,
        "decay_factor": 0.5,
        "trunk_lr_scale": 0.1,
        "ema_decay": 0.9,
        "grad_clip_max_norm": 1.0,
    }
    base.update(kw)
    return TrainConfig(**base)


def _grads(model, step, scale) -> None:
    gen = torch.Generator().manual_seed(step)
    for n, p in model.named_parameters():
        if p.requires_grad and not n.startswith("unused"):
            p.grad = torch.randn(p.shape, generator=gen) * scale


def _rel(a, b) -> float:
    return ((a - b).abs() / (b.abs() + 1e-12)).max().item()


@pytest.mark.parametrize("scale", [0.01, 5.0], ids=["clip_inactive", "clip_active"])
def test_fused_step_equals_the_reference_over_several_steps(scale):
    cfg = _cfg()
    ref, fus = Tiny(), Tiny()
    fus.load_state_dict(ref.state_dict())
    opt_ref = optim.build_optimizer(ref, cfg)
    ema_ref = optim.EMA(ref, cfg.ema_decay)
    f = fused_opt.build_fused(fus, cfg)
    assert {g["name"] for g in f.optimizer.param_groups} == {"heads", "trunk"}
    params_ref = [p for g in opt_ref.param_groups for p in g["params"]]
    assert len(f.params) == len(params_ref)
    assert all(
        not n.startswith("frozen")
        for n, p in fus.named_parameters()
        if any(p is q for q in f.params)
    )
    for step in range(8):
        lr_ref = optim.set_lr(opt_ref, step, cfg)
        lr_fus = f.set_lr(step, cfg)
        assert lr_ref == lr_fus
        assert [g["lr"] for g in opt_ref.param_groups] == [
            g["lr"] for g in f.optimizer.param_groups
        ]
        _grads(ref, step, scale)
        _grads(fus, step, scale)
        norm_ref = optim.clip_grad_norm(params_ref, cfg.grad_clip_max_norm)
        opt_ref.step()
        ema_ref.update(ref)
        norm_fus = f.step()
        assert abs(norm_ref.item() - norm_fus.item()) <= 1e-6 * norm_ref.item()
        for (n, a), (_, b) in zip(
            ref.named_parameters(), fus.named_parameters(), strict=False
        ):
            assert _rel(b, a) <= 1e-6, n
            assert _rel(f.ema.shadow[n], ema_ref.shadow[n]) <= 1e-6, n
        for pa, pb in zip(params_ref, f.params, strict=False):
            sa, sb = opt_ref.state[pa], f.optimizer.state[pb]
            assert bool(sa) == bool(sb)  # grad None: no state in either
            if not sa:
                continue
            assert _rel(sb["exp_avg"], sa["exp_avg"]) <= 1e-6
            assert _rel(sb["exp_avg_sq"], sa["exp_avg_sq"]) <= 1e-6
            assert float(sb["step"]) == float(sa["step"])
        opt_ref.zero_grad(set_to_none=True)
        f.zero_grad()
    assert fus.frozen.weight.requires_grad is False
    assert torch.equal(fus.frozen.weight, ref.frozen.weight)
    assert float(norm_fus) > 0.0


def test_clip_scale_matches_and_clips_to_max_norm():
    model = Tiny()
    _grads(model, 0, 5.0)
    params = [p for p in model.parameters() if p.grad is not None]
    ref = copy.deepcopy([p.grad for p in params])
    total_ref = optim.clip_grad_norm(params, 1.0)
    clipped_ref = [p.grad.clone() for p in params]
    for p, g in zip(params, ref, strict=False):
        p.grad = g.clone()
    total = fused_opt.foreach_clip_grad_norm(params, 1.0)
    assert total_ref.item() > 1.0
    assert abs(total.item() - total_ref.item()) <= 1e-6 * total_ref.item()
    for p, c in zip(params, clipped_ref, strict=False):
        assert _rel(p.grad, c) <= 1e-6
    assert torch.linalg.vector_norm(
        torch.stack([p.grad.norm() for p in params])
    ).item() == pytest.approx(1.0, rel=1e-4)


def test_no_gradients_returns_zero_and_leaves_parameters():
    model = Tiny()
    before = copy.deepcopy(model.state_dict())
    f = fused_opt.build_fused(model, _cfg())
    f.set_lr(5)
    assert f.step().item() == 0.0
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k])


def test_fused_set_lr_matches_the_schedule():
    cfg = _cfg(
        max_lr=1e-2, min_lr=1e-3, warmup_steps=4, decay_steps=1, decay_factor=0.1
    )
    f = fused_opt.build_fused(Tiny(), cfg)
    for step, expect in [(2, 5e-3), (4, 1e-2), (40, 1e-3)]:
        assert f.set_lr(step) == pytest.approx(expect)
    assert {g["name"]: g["lr"] for g in f.optimizer.param_groups}[
        "trunk"
    ] == pytest.approx(1e-4)


def test_ema_off_and_bad_impl():
    f = fused_opt.build_fused(Tiny(), _cfg(use_ema=False))
    assert f.ema is None
    with pytest.raises(ValueError, match="impl must be"):
        fused_opt.build_fused(Tiny(), _cfg(), impl="nope")
