"""Tests for phase_split.py: segment times from SMF_BENCH_PHASE_TRACE marks."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import phase_split  # noqa: E402

LINES = """\
step 0 sub_ms=1.0
# seg fi_entry t=1.0000000
# seg fi_mapped t=1.0001000
# seg fi_call t=1.0002000
# seg mf_setup t=1.0003000
# seg mf_p2 t=1.0005000
# seg fi_end t=1.0010000
noise line
# seg fi_entry t=2.0000000
# seg fi_mapped t=2.0002500
# seg fi_call t=2.0003000
# seg mf_setup t=2.0004000
# seg fi_end t=2.0009000
""".splitlines()


def test_segments_per_factorization():
    f = phase_split.split(LINES)
    assert len(f) == 2
    assert f[0]["fi_entry->fi_mapped"] == pytest.approx(0.1)
    assert f[1]["fi_entry->fi_mapped"] == pytest.approx(0.25)
    assert f[0]["mf_setup->mf_p2"] == pytest.approx(0.2)


def test_missing_mark_keeps_other_segments():
    f = phase_split.split(LINES)
    assert "mf_setup->mf_p2" not in f[1]
    assert f[1]["mf_setup->fi_end"] == pytest.approx(0.5)
    assert f[1]["fi_entry->fi_end"] == pytest.approx(0.9)


def test_table_medians(tmp_path):
    (tmp_path / "phase_caseA_T8_0.txt").write_text("\n".join(LINES), encoding="utf-8")
    rows = phase_split.medians(tmp_path)
    assert rows[("caseA", 8)]["fi_entry->fi_mapped"] == pytest.approx(0.175)


def test_warmup_files_ignored(tmp_path):
    (tmp_path / "phase_caseA_T8_0.txt").write_text("\n".join(LINES), encoding="utf-8")
    slow = [ln.replace("t=1.0001000", "t=1.9") for ln in LINES]
    (tmp_path / "phase_caseA_T8_w.txt").write_text("\n".join(slow), encoding="utf-8")
    assert phase_split.medians(tmp_path)[("caseA", 8)]["fi_entry->fi_mapped"] == pytest.approx(0.175)


def test_sum_lines_join_their_factorization():
    lines = ["# seg fi_entry t=1.000", "# seg mf_p2 t=1.001", "# sum top_asm s=0.0005",
             "# sum top_panel s=0.0002", "# seg mf_p3 t=1.004", "# seg fi_end t=1.005",
             "# seg fi_entry t=2.000", "# seg fi_end t=2.001"]
    out = phase_split.split(lines)
    assert out[0]["sum:top_asm"] == pytest.approx(0.5)
    assert out[0]["sum:top_panel"] == pytest.approx(0.2)
    assert "sum:top_asm" not in out[1]
