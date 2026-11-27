"""AlphaFold3 ranking in the output report and in FoldBench ranking.csv rows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from foldjet.eval import af3_ranking as R

ROOT = Path(__file__).resolve().parents[2]


def _fake_terms(cif, ptm, iptm) -> R.RankingTerms:
    if cif == "bad":
        msg = "unparseable"
        raise ValueError(msg)
    fd = {"a": 0.1, "b": 0.3, "c": 0.3}[cif]
    single = iptm is None
    score = (ptm if single else 0.8 * iptm + 0.2 * ptm) + 0.5 * fd
    return R.RankingTerms(
        ptm, None if single else iptm, fd, has_clash=False, ranking_score=score
    )


def test_rank_samples_best_first_on_tie_and_errors_kept(monkeypatch):
    monkeypatch.setattr(R, "ranking_terms", _fake_terms)
    rows, best = R.rank_samples(["a", "b", "bad", "c"], [0.5, 0.5, 0.9, 0.5], None)
    assert best == 1  # b and c tie: the first one (AlphaFold3 / pandas idxmax order)
    assert "error" in rows[2]
    assert rows[1]["ranking_score"] == pytest.approx(0.65)
    rows, best = R.rank_samples(["bad"], [0.5], [0.4])
    assert best is None


def test_foldbench_rows_use_af3_score_and_fall_back_on_error(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "benchmarks" / "foldbench"))
    import fb_io

    monkeypatch.setattr(R, "ranking_terms", _fake_terms)
    monkeypatch.setattr(fb_io, "rename_chains", lambda cif, names: cif)
    rows = fb_io.write_seed(
        str(tmp_path),
        "t",
        7,
        ["a", "bad"],
        [0.6, 0.7],
        ["A"],
        [0.5, 0.9],
        [0.625, 0.65],
    )
    assert len(fb_io.RANKING_HEADER.split(",")) == len(rows[0])
    assert rows[0][2] == pytest.approx(0.8 * 0.625 + 0.2 * 0.5 + 0.05)
    assert rows[0][3] == 0.6
    assert rows[1][2] == 0.7
    assert rows[1][-1].startswith("ValueError")
    plain = fb_io.write_seed(str(tmp_path), "t", 7, ["a"], [0.6], ["A"])
    assert plain[0][2] == plain[0][3] == 0.6
    assert (tmp_path / "t_seed7_sample1_pred.cif").read_text() == "bad"
