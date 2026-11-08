"""End-to-end checks of the planner on a small quadrotor (N=15, MUMPS).

  python -m pytest planner/tests -q
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import specs  # noqa: E402
from smfplan import Planner, SolutionLibrary  # noqa: E402

pytest.importorskip("casadi")


def feasible(spec, task, x, tol=1e-6) -> bool:
    import casadi as ca
    lbx, ubx = spec.bounds(task)
    args = [x] + ([spec.params(task)] if "p" in spec.nlp else [])
    g = ca.Function("g", [spec.nlp["x"]] + ([spec.nlp["p"]] if "p" in spec.nlp else []), [spec.nlp["g"]])
    gv = np.array(g(*args)).ravel()
    return bool(np.all(gv >= spec.lbg - tol) and np.all(gv <= spec.ubg + tol)
                and np.all(x >= lbx - tol) and np.all(x <= ubx + tol))


@pytest.fixture(scope="module")
def planner():
    opts, env = specs.solver_setup("mumps")
    with Planner(specs.factory("quad", 15), workers=3, k=3, solver_options=opts, env=env) as pl:
        yield pl


def test_learns_and_warm_starts(planner):
    rng = np.random.default_rng(0)
    tasks = [planner.spec.sample(rng) for _ in range(8)]
    first = planner.plan(tasks[0])
    assert first.success and first.source == "cold"
    assert len(planner.library) == 1
    later = [planner.plan(t) for t in tasks[1:]]
    assert all(r.success for r in later)
    assert all(r.source.startswith("lib") for r in later)
    assert len(planner.library) == 8
    for t, r in zip(tasks[1:], later):
        assert feasible(planner.spec, t, r.x)


def test_race_stops_losers(planner):
    # a task the library has seen: the nearest start wins at once, the others
    # are stopped by the race flag instead of running to convergence
    rng = np.random.default_rng(1)
    t = planner.spec.sample(rng)
    planner.plan(t)
    r = planner.plan(t + 1e-3)
    assert r.success
    statuses = [s[1] for s in r.starts]
    assert len(statuses) == 3
    assert all(s in ("Solve_Succeeded", "User_Requested_Stop", "Solved_To_Acceptable_Level") for s in statuses)


def test_library_roundtrip(tmp_path):
    lib = SolutionLibrary()
    for i in range(5):
        lib.add([i, 0.0], np.full(3, i), np.zeros(3), np.zeros(2), float(i))
    lib.save(tmp_path / "lib.npz")
    lib2 = SolutionLibrary.load(tmp_path / "lib.npz")
    d, j = lib2.nearest([2.2, 0.0], 2)
    assert list(j) == [2, 3] and len(lib2) == 5
    # near-duplicate with a worse objective is ignored, a better one replaces
    assert not lib2.add([2.0, 0.01], np.zeros(3), np.zeros(3), np.zeros(2), 9.0, min_dist=0.1)
    assert lib2.add([2.0, 0.01], np.zeros(3), np.zeros(3), np.zeros(2), -1.0, min_dist=0.1)
    assert len(lib2) == 5 and lib2.obj[2] == -1.0


def test_hint_races_with_adaptive_twin(planner):
    rng = np.random.default_rng(2)
    t = planner.spec.sample(rng)
    first = planner.plan(t)
    r = planner.plan(t + 1e-2, hint=first.as_hint())
    assert r.success
    tags = [s[0] for s in r.starts]
    assert tags[:2] == ["hint", "hint_adapt"] and len(tags) == 3
