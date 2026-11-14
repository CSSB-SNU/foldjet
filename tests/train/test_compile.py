"""CPU tests of ``foldforge.train.compile``: which modules are wrapped, idempotence, forward equivalence through ``torch.compile`` (a toy block named like the
engine class; the real blocks are exercised by ``scripts/train_protenix1.py --bench --compile``), the strict (no recompile) guard and the step-graph config key."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from foldforge.train import compile as train_compile
from foldforge.train.config import TrainConfig


class PairformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x, scale=1.0):
        return torch.relu(self.linear(x)) * scale


class Other(nn.Module):
    def forward(self, x):
        return x


def toy():
    return nn.Sequential(PairformerBlock(), Other(), PairformerBlock())


def test_wraps_only_the_region_classes_once():
    model = toy()
    assert train_compile.enable_compile(model, "blocks") == 2
    assert train_compile.enable_compile(model, "blocks") == 0  # idempotent
    assert not hasattr(model[1], "_ff_eager_forward")
    assert train_compile.enable_compile(toy(), "diffusion") == 0


def test_compiled_forward_matches_eager_and_does_not_recompile_across_instances():
    torch.manual_seed(0)
    model, reference = toy(), toy()
    reference.load_state_dict(model.state_dict())
    train_compile.enable_compile(model, "blocks")
    x = torch.randn(3, 4)
    expected = reference(x)
    torch._dynamo.reset()  # noqa: SLF001
    first = model(x)
    with train_compile.strict():  # both instances share ONE compiled graph: the second call compiles nothing
        again = model(x)
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(again, expected)
    assert train_compile.report()["graph_breaks"] == {}


def test_strict_refuses_a_new_shape():
    model = toy()
    train_compile.enable_compile(model, "blocks")
    torch._dynamo.reset()  # noqa: SLF001
    model(torch.randn(3, 4))
    with train_compile.strict(), pytest.raises(Exception, match="Recompiling"):
        model(torch.randn(5, 4))


def test_compile_is_part_of_the_step_graph_key():
    cfg = TrainConfig()
    assert cfg.compile == ""
    assert replace(cfg, compile="all") != cfg  # the step graph is keyed on the config: eager and compiled captures never mix
