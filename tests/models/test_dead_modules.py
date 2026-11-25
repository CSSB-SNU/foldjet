"""``modules.dense.dead``: numerically dead residual modules are flagged.

The numerically dead residual modules of a checkpoint are flagged once the weights
are final.
"""

import torch
from torch import nn

from foldjet.modules.dense.dead import THRESHOLD, clear_dead, is_dead, mark_dead


class Residual(nn.Module):
    """Stands for an engine residual module.

    The class declares the plain ``dead`` flag the engine honours.
    """

    dead = False

    def __init__(self, fill: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((4, 4), fill))
        self.bias = nn.Parameter(torch.full((4,), fill))


class Plain(nn.Module):
    """A module that does not declare the flag: never flagged.

    The engine decides which of its modules honour it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))


def _model() -> nn.Module:
    model = nn.Module()
    model.live = Residual(0.5)
    model.dead_a = Residual(5e-37)
    model.dead_b = Residual(0.0)
    model.plain = Plain()
    return model


def test_is_dead_needs_every_parameter_below_the_threshold():
    mixed = Residual(5e-37)
    with torch.no_grad():
        mixed.bias[0] = 1e-3
    assert not is_dead(mixed)
    assert is_dead(Residual(5e-37))
    assert not is_dead(Residual(THRESHOLD * 10))
    assert not is_dead(nn.Module()), "a module without parameters is not dead"


def test_mark_dead_flags_only_dead_residual_modules_and_counts_by_class():
    model = _model()
    assert mark_dead(model) == {"Residual": 2}
    assert not model.live.dead
    assert model.dead_a.dead
    assert model.dead_b.dead
    assert not hasattr(model.plain, "dead"), (
        "a class that does not declare the flag is left alone, dead weights or not"
    )


def test_mark_dead_clears_a_flag_whose_weights_came_back():
    model = _model()
    mark_dead(model)
    with torch.no_grad():
        model.dead_a.weight.fill_(0.25)
        model.dead_a.bias.fill_(0.25)
    assert mark_dead(model) == {"Residual": 1}
    assert not model.dead_a.dead


def test_clear_dead_runs_every_module_again_and_counts_what_was_flagged():
    model = _model()
    mark_dead(model)
    assert clear_dead(model) == 2
    assert not any(m.dead for m in model.modules() if isinstance(m, Residual))
    assert clear_dead(model) == 0


def test_the_flag_is_a_plain_attribute_so_compile_guards_on_it():
    module = Residual(5e-37)
    mark_dead(module)
    assert "dead" in vars(module)
    assert module.dead is True


# --- the engine contract `mark_dead` relies on --------------------------------------
# (the engine declined to ship its own loader: FoldJet owns this)


def test_the_engine_residual_classes_still_declare_the_dead_flag_false_by_default():
    from miniworld_engine.modules.attention import AttentionPairBias
    from miniworld_engine.modules.msa import MSAPairWeightedAveraging, OuterProductMean
    from miniworld_engine.modules.transition import Transition
    from miniworld_engine.modules.triangle import (
        TriangleAttention,
        TriangleMultiplication,
    )

    for cls in (
        Transition,
        AttentionPairBias,
        TriangleAttention,
        TriangleMultiplication,
        OuterProductMean,
        MSAPairWeightedAveraging,
    ):
        assert hasattr(cls, "dead"), (
            f"{cls.__name__} lost the `dead` flag: "
            "dead-module skipping would silently stop (speed only)"
        )
        assert cls.dead is False, f"{cls.__name__}.dead must default to False"


def test_a_flagged_engine_transition_returns_its_residual_input_without_grad():
    from miniworld_engine.modules.transition import Transition

    module = Transition(d_hidden=16, n=2, backend="pytorch").eval()
    with torch.no_grad():
        for p in module.parameters():
            p.fill_(5e-37)
    assert mark_dead(module) == {"Transition": 1}
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(module(x), x), (
            "a dead Transition must return x unchanged (x + 0)"
        )
    module.dead = False
    with torch.no_grad():
        assert torch.allclose(module(x).float(), x.float(), atol=1e-2)
