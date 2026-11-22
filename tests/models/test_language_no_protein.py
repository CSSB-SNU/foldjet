"""An input without protein tokens (RNA / DNA / ligand only): refused by default, served as zero hidden states with ``allow_no_protein``.

Regression for the ESMFold2 FoldBench runs on nucleic-acid-only targets (`No protein token reached the language model`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from foldforge.modules import language_model as lm

N_LAYERS, D = 3, 8


class _FakeLM:
    """Returns N_LAYERS + 1 hidden-state tensors of ones over whatever sequence it is given."""

    def __init__(self):
        self.calls = []

    def __call__(self, ids, *, sequence_id, output_hidden_states):
        self.calls.append(ids.shape)
        b, w = ids.shape
        return SimpleNamespace(hidden_states=tuple(torch.ones(b, w, D) for _ in range(N_LAYERS + 1)))


def _inputs(batch: int = 1, length: int = 6):
    input_ids = torch.full((batch, length), 5, dtype=torch.long)
    asym_id = torch.zeros(batch, length, dtype=torch.long)
    residue_index = torch.arange(length).expand(batch, length).clone()
    is_protein = torch.zeros(batch, length, dtype=torch.bool)  # RNA only
    mask = torch.ones(batch, length, dtype=torch.bool)
    return input_ids, asym_id, residue_index, is_protein, mask


def test_no_protein_input_is_refused_by_default():
    with pytest.raises(ValueError, match="No protein token"):
        lm.compute_lm_hidden_states(_FakeLM(), *_inputs())


@pytest.mark.parametrize("pad", [None, 8])
def test_no_protein_input_gives_zero_hidden_states_when_allowed(pad):
    fake = _FakeLM()
    out = lm.compute_lm_hidden_states(fake, *_inputs(batch=2, length=6), allow_no_protein=True, pad_to_multiple=pad)
    assert out.shape == (2, 6, N_LAYERS + 1, D)
    assert torch.count_nonzero(out) == 0
    assert len(fake.calls) == 1 and fake.calls[0][1] == (pad or 2)  # one minimal BOS/EOS call for the layer count / width


def test_allow_no_protein_does_not_change_a_protein_input():
    ids, asym, res, _, mask = _inputs()
    is_protein = torch.ones_like(mask)
    base = lm.compute_lm_hidden_states(_FakeLM(), ids, asym, res, is_protein, mask)
    allowed = lm.compute_lm_hidden_states(_FakeLM(), ids, asym, res, is_protein, mask, allow_no_protein=True)
    assert torch.equal(base, allowed)
