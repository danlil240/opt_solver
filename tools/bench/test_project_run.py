"""Tests for project_run.py: projected IPOPT-run factorization time and the stage gates."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import project_run  # noqa: E402


def rec(matrix, threads, solver, analyse_ms, factor_ms, refactor_ms, batch_count,
        warmup=False, outcome="Success"):
    return {"matrix_id": matrix, "threads": threads, "solver_id": solver,
            "analyse_ms": analyse_ms, "factor_ms": factor_ms, "refactor_ms": refactor_ms,
            "batch_count": batch_count, "outcome": outcome, "warmup": warmup,
            "peak_rss_bytes": 1 << 20}


def test_projection_formula():
    recs = [rec("c", 8, "smf", analyse_ms=10, factor_ms=50, refactor_ms=4, batch_count=5),
            rec("c", 8, "smf", analyse_ms=12, factor_ms=54, refactor_ms=6, batch_count=5)]
    p = project_run.project(recs, {"c": 25})
    assert p[("c", 8, "smf")] == 11 + 52 + 20 * 5


def test_warmups_ignored_and_gate():
    recs = [rec("a", 8, "smf", 1, 1, 1, 1), rec("a", 8, "mkl_pardiso", 1, 2, 2, 1),
            rec("a", 8, "smf", 100, 100, 100, 1, warmup=True)]
    r = project_run.ratios(project_run.project(recs, {"a": 3}), 8)
    assert r["a"] == pytest.approx(4 / 7)
    assert project_run.gate_failures(r, {"a"}, recs) == []
    assert project_run.gate_failures({"a": 1.0}, {"a"}, recs) == ["a"]


def test_gate_fails_on_inaccurate_smf_and_missing_case():
    recs = [rec("a", 8, "smf", 1, 1, 1, 1, outcome="Inaccurate"),
            rec("a", 8, "mkl_pardiso", 1, 2, 2, 1)]
    r = project_run.ratios(project_run.project(recs, {"a": 3}), 8)
    assert project_run.gate_failures(r, {"a"}, recs) == ["a"]
    assert project_run.gate_failures(r, {"a", "b"}, [rec("a", 8, "smf", 1, 1, 1, 1)]) == ["b"]


def test_gate_sets():
    assert len(project_run.STAGE1_WINS) == 6
    assert project_run.STAGE1_WINS < project_run.STAGE2_WINS
    assert len(project_run.STAGE2_WINS) == 8
