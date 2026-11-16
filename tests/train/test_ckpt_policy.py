"""``prepare.enable_activation_checkpointing`` policies: how many blocks each wraps on a fake model (CPU)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from foldforge.train import prepare


class PairformerBlock(nn.Module):
    def forward(self, x):
        return x * 2


class EvoformerBlock(PairformerBlock):
    pass


class Fake(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk_pairformer = nn.ModuleList(PairformerBlock() for _ in range(8))
        self.msa_stack = nn.ModuleList(EvoformerBlock() for _ in range(4))
        self.tmpl_pairformer = nn.ModuleList(PairformerBlock() for _ in range(2))
        self.confidence_pairformer = nn.ModuleList(PairformerBlock() for _ in range(4))


def wrapped(policy: str | None) -> tuple[int, set[str]]:
    model = Fake()
    count = (
        prepare.enable_activation_checkpointing(model)
        if policy is None
        else prepare.enable_activation_checkpointing(model, policy)
    )
    return count, {
        n for n, m in model.named_modules() if getattr(m, "_ff_checkpointed", False)
    }


@pytest.mark.parametrize(
    ("policy", "count"),
    [
        (None, 18),  # the old signature: every block
        ("all", 18),
        ("none", 0),
        ("pairformer", 8),
        ("evoformer", 4),
        ("except:trunk", 10),
        ("except:msa", 14),
        ("every:2", 4 + 2 + 4 + 4),
        ("every:3", 3 + 10),
        ("every:1", 18),
        ("first:3", 3 + 10),
        ("last:3", 3 + 10),
        ("last:99", 18),
    ],
)
def test_counts(policy, count):
    assert wrapped(policy)[0] == count


def test_positions():
    _, names = wrapped("every:4")
    assert {
        f"trunk_pairformer.{i}" for i in (0, 4)
    } <= names and "trunk_pairformer.1" not in names
    _, names = wrapped("last:2")
    assert {
        "trunk_pairformer.6",
        "trunk_pairformer.7",
    } <= names and "trunk_pairformer.5" not in names


def test_unknown_policy():
    with pytest.raises(ValueError):
        wrapped("bogus")
    with pytest.raises(ValueError):
        wrapped("every:0")


def test_wrapped_forward_still_correct():
    model = Fake()
    prepare.enable_activation_checkpointing(model, "pairformer")
    x = torch.ones(3, requires_grad=True)
    model.trunk_pairformer[0](x).sum().backward()
    assert torch.equal(x.grad, torch.full((3,), 2.0))
