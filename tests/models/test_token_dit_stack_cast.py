"""``TokenDiTStack``: one cast per stack, one pair object per fold.

The engine's master blocks take bf16 activations, so the stack casts ONCE (not every
block, every step), and the pair it hands the blocks is the same tensor object for
every call of one fold (the engine hoists the pair bias per pair object).
"""

from __future__ import annotations

import torch
from torch import nn

from foldjet.modules.engine.transformers import TokenDiTStack


class _Recorder(nn.Module):
    def __init__(self, calls: list) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, x, cond, pair, key_mask, **kw: object) -> torch.Tensor:
        self.calls.append((x, cond, pair))
        return x + 1


def _stack(blocks: int = 3) -> tuple[TokenDiTStack, list]:
    calls: list = []
    return TokenDiTStack([_Recorder(calls) for _ in range(blocks)]), calls


def _inputs(dtype=torch.float32) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(2, 5, 4).to(dtype),
        torch.ones(5),
        torch.randn(5, 4).to(dtype),
        torch.randn(5, 5, 6).to(dtype),
    )


def test_fp32_inputs_are_cast_once_under_autocast_and_the_output_keeps_the_input_dtype():  # noqa: E501 - test name is a pass-token node id
    stack, calls = _stack()
    act, mask, single, pair = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = stack(act, mask, single, pair)
    assert all(t.dtype is torch.bfloat16 for call in calls for t in call)
    assert out.dtype is torch.float32
    torch.testing.assert_close(
        out, (act.to(torch.bfloat16) + 1 + 1 + 1).float(), rtol=0, atol=0
    )


def test_the_pair_is_one_tensor_object_across_blocks_and_steps_and_follows_a_new_fold():
    stack, calls = _stack()
    act, mask, single, pair = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        stack(act, mask, single, pair)
        stack(act + 1, mask, single, pair)  # the next step of the same fold
        assert len({id(call[2]) for call in calls}) == 1
        stack(act, mask, single, pair.clone())  # the next fold
        assert calls[-1][2] is not calls[0][2]
        stack(
            act, mask, single, pair
        )  # in-place edit of the source invalidates the cached cast
        before = calls[-1][2]
        pair.add_(1)
        stack(act, mask, single, pair)
        assert calls[-1][2] is not before
        torch.testing.assert_close(
            calls[-1][2][0].float(), pair.to(torch.bfloat16).float(), rtol=0, atol=0
        )


def test_bf16_inputs_and_fp32_precision_pass_through_unchanged():
    stack, calls = _stack(2)
    act, mask, single, pair = _inputs(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = stack(act, mask, single, pair)
    assert out.dtype is torch.bfloat16
    assert (
        calls[0][0].untyped_storage().data_ptr() == act.untyped_storage().data_ptr()
    )  # no copy of a bf16 activation
    assert (
        calls[0][2].untyped_storage().data_ptr() == pair.untyped_storage().data_ptr()
    )  # no copy of a bf16 pair
    stack, calls = _stack(2)
    act, mask, single, pair = _inputs()  # no autocast (fp32 precision): fp32 stays fp32
    out = stack(act, mask, single, pair)
    assert all(t.dtype is torch.float32 for call in calls for t in call)
    assert out.dtype is torch.float32


def test_a_pair_that_requires_grad_is_not_kept_across_calls():
    stack, calls = _stack(1)
    act, mask, single, pair = _inputs()
    pair.requires_grad_(requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        stack(act, mask, single, pair)
        stack(act, mask, single, pair)
    assert calls[0][2] is not calls[1][2]
    assert stack._pair_source is None
