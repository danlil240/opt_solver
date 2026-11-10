"""Tests for fit_front_costs.py: per-front cost constants of the thread plan."""

from __future__ import annotations

import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import fit_front_costs  # noqa: E402

A, B, C, FPRINTF = 1e-10, 3e-9, 2e-6, 5e-7


def synthetic(n=4000, seed=3):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        f = rng.randint(1, 400)
        p = rng.randint(1, f)
        out.append((f, p, p, A * f * f * p + B * f * f + C))
    return out


def test_fit_recovers_constants():
    c_asm, c_front = fit_front_costs.fit(synthetic(), FPRINTF)
    assert c_asm == pytest.approx(30.0, rel=0.01)
    assert c_front == pytest.approx(15000.0, rel=0.01)


def test_parse_trace(tmp_path):
    t = tmp_path / "fronts_x_T1_trace_1.txt"
    t.write_text(
        "# cmd=...\n"
        "# phase f ne p me blas tiled lapack flops ta tk0 tk1 s parent psym panel defer schur stall nstall nfull\n"
        "# factor t=0.0 work_limit=-1 static=0\n"
        "0 10 4 3 0 1 0 0 1e3 1.000 1.0001 1.0002 0 2 3 0 0 0 0 0 0\n"
        "0 20 5 5 0 1 0 0 1e3 1.004 1.0041 1.0045 1 2 5 0 0 0 0 0 0\n"
        "# factor t=2.0 work_limit=-1 static=0\n"
        "0 10 4 4 0 1 0 0 1e3 2.000 2.0001 2.0003 0 2 4 0 0 0 0 0 0\n", encoding="utf-8")
    fronts = fit_front_costs.parse(t)
    # (f, ne, p, seconds): next front's ta - ta; a factorization's last front: tk1 - ta
    assert fronts[0][:3] == (10, 4, 3) and fronts[0][3] == pytest.approx(0.004)
    assert fronts[1][:3] == (20, 5, 5) and fronts[1][3] == pytest.approx(0.0005)
    assert fronts[2][3] == pytest.approx(0.0003)
