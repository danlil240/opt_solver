"""Tests for tools/bench/score.py on synthetic records."""

from __future__ import annotations

import csv
import unittest
import io
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import score  # noqa: E402


def rec(matrix, solver, wall, *, track="recommended", eta=1e-15, omega=1e-14,
        outcome="Success", threads=1, inertia=None, **extra):
    r = {
        "matrix_id": matrix, "solver_id": solver, "track": track,
        "threads": threads, "warmup": False, "outcome": outcome,
        "mode": "cold", "wall_ms": wall,
        "analyse_ms": wall * 0.2, "factor_ms": wall * 0.6,
        "solve_ms": wall * 0.1, "refine_ms": wall * 0.1,
        "validation": {"finite": True, "eta_inf": eta, "omega": omega},
        "inertia": inertia or {"pos": 6, "neg": 4, "zero": 0},
        "n": 10,
    }
    r.update(extra)
    return r


def case(matrix, cls, split, track="recommended"):
    return {"matrix_id": matrix, "class": cls, "split": split, "track": track,
            "matrix_type": "indefinite"}


def manifest(cases):
    return {"cases": cases}


class TestScore(unittest.TestCase):
    def test_shifted_geomean_uses_best_accurate(self):
        m = manifest([case("a", "hard_indef", "tuning")])
        records = [
            rec("a", "smf", 100.0),
            rec("a", "ma27", 200.0),
            # faster but inaccurate: must not be "best"
            rec("a", "mkl_pardiso", 10.0, eta=1e-5),
            # smf itself is excluded from the competitor set even if it is fastest
        ]
        res = score.score(records, m, split="tuning", shift_ms=10.0)
        row = res["hard_indef"]["recommended"][1]["ratio"]
        self.assertEqual(row["n"], 1)
        self.assertTrue(math.isclose(row["geomean"], (100.0 + 10.0) / (200.0 + 10.0)))
        self.assertEqual(row["best_solver_counts"], {"ma27": 1})


    def test_split_filter(self):
        m = manifest([case("a", "spd", "tuning"), case("b", "spd", "heldout")])
        records = [rec("a", "smf", 10.0), rec("a", "ma27", 20.0),
                   rec("b", "smf", 40.0), rec("b", "ma27", 20.0)]
        tuning = score.score(records, m, split="tuning", shift_ms=10.0)
        held = score.score(records, m, split="heldout", shift_ms=10.0)
        both = score.score(records, m, split="all", shift_ms=10.0)
        self.assertEqual(tuning["spd"]["recommended"][1]["ratio"]["n"], 1)
        self.assertTrue(math.isclose(tuning["spd"]["recommended"][1]["ratio"]["geomean"], 20 / 30))
        self.assertEqual(held["spd"]["recommended"][1]["ratio"]["n"], 1)
        self.assertTrue(math.isclose(held["spd"]["recommended"][1]["ratio"]["geomean"], 50 / 30))
        self.assertEqual(both["spd"]["recommended"][1]["ratio"]["n"], 2)


    def test_failure_counts_by_class(self):
        m = manifest([
            case("a", "small_kkt", "tuning"), case("b", "small_kkt", "tuning"),
            case("c", "small_kkt", "tuning"), case("d", "spd", "tuning"),
        ])
        records = [
            rec("a", "smf", 5.0), rec("a", "ma27", 6.0),
            rec("b", "smf", 5.0, eta=1e-6),          # accuracy failure
            rec("b", "ma27", 6.0),
            rec("c", "smf", 0.0, outcome="Failed"),  # hard failure
            rec("c", "ma27", 6.0),
            rec("d", "smf", 5.0), rec("d", "ma27", 6.0, omega=1e-5),  # ma27 inaccurate
        ]
        res = score.score(records, m, split="all", shift_ms=10.0)
        s = res["small_kkt"]["recommended"][1]["solvers"]["smf"]
        self.assertEqual((s["total"], s["solved"], s["inaccurate"], s["failed"]), (3, 1, 1, 1))
        s27 = res["spd"]["recommended"][1]["solvers"]["ma27"]
        self.assertEqual((s27["total"], s27["solved"], s27["inaccurate"], s27["failed"]), (1, 0, 1, 0))
        self.assertEqual(res["spd"]["recommended"][1]["solvers"]["smf"]["solved"], 1)


    def test_profile_csv_monotone(self):
        times = {  # solver -> case -> accurate time (None = unsolved)
            "smf": {"a": 10.0, "b": 30.0, "c": None},
            "ma27": {"a": 20.0, "b": 10.0, "c": 5.0},
        }
        taus = [1.0, 1.5, 2.0, 3.0, 10.0]
        rows = score.performance_profile(times, taus)
        self.assertEqual([r["tau"] for r in rows], taus)
        for solver in times:
            vals = [r[solver] for r in rows]
            self.assertEqual(vals, sorted(vals))
            self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))
        self.assertTrue(rows[0]["smf"] == 1 / 3 and rows[0]["ma27"] == 2 / 3)
        self.assertEqual(rows[-1]["smf"], 2 / 3)  # case c never solved by smf
        buf = io.StringIO()
        score.write_profile_csv(rows, ["smf", "ma27"], buf)
        parsed = list(csv.DictReader(io.StringIO(buf.getvalue())))
        self.assertEqual(list(parsed[0].keys()), ["tau", "smf", "ma27"])
        self.assertEqual(len(parsed), len(taus))


    def test_inertia_column_on_ipopt_track_only(self):
        m = manifest([case("a", "hard_indef", "tuning", track="ipopt"),
                      case("a", "hard_indef", "tuning", track="recommended")])
        records = [
            rec("a", "smf", 5.0, track="ipopt", inertia={"pos": 5, "neg": 5, "zero": 0}),
            rec("a", "ma27", 6.0, track="ipopt"),
            rec("a", "ma57", 6.0, track="ipopt"),
            rec("a", "smf", 5.0), rec("a", "ma27", 6.0),
        ]
        res = score.score(records, m, split="all", shift_ms=10.0)
        ip = res["hard_indef"]["ipopt"][1]["solvers"]
        self.assertTrue(ip["smf"]["inertia_pass"] == 0 and ip["smf"]["inertia_checked"] == 1)
        self.assertEqual(ip["ma27"]["inertia_pass"], 1)
        rc = res["hard_indef"]["recommended"][1]["solvers"]
        self.assertEqual(rc["smf"]["inertia_checked"], 0)


    def test_thread_groups_are_separate_tables(self):
        m = manifest([case("a", "spd", "tuning")])
        records = [rec("a", "smf", 10.0), rec("a", "ma27", 20.0),
                   rec("a", "smf", 4.0, threads=8), rec("a", "mkl_pardiso", 8.0, threads=8)]
        res = score.score(records, m, split="all", shift_ms=10.0)
        self.assertEqual(set(res["spd"]["recommended"]), {1, 8})
        self.assertTrue(math.isclose(res["spd"]["recommended"][8]["ratio"]["geomean"], 14 / 18))


    def test_phase_columns_warm_sequence_fallback(self):
        m = manifest([case("a", "large_kkt", "tuning", track="ipopt")])
        r = rec("a", "smf", 100.0, track="ipopt", mode="warm_sequence", batch_count=5,
                factor_ms=50.0)
        res = score.score([r, rec("a", "ma27", 90.0, track="ipopt")], m, split="all")
        ph = res["large_kkt"]["ipopt"][1]["solvers"]["smf"]["phases"]
        self.assertTrue(math.isclose(ph["refactor_ms"], 10.0))
        self.assertTrue(ph["first_factor_ms"] is None)


    def test_replay_override_args(self):
        import run
        self.assertEqual(run.replay_override_args({"ordering": "matching"}), ["--ordering", "matching"])
        self.assertEqual(run.replay_override_args({}), [])

    def test_report_only_excluded_from_gates_ratio_and_listed(self):
        m = manifest([case("a", "hard_indef", "tuning"),
                      dict(case("sing", "hard_indef", "tuning"), report_only=True)])
        records = [rec("a", "smf", 100.0), rec("a", "ma27", 200.0),
                   rec("sing", "smf", 5.0, eta=1e-3), rec("sing", "ma27", 6.0)]
        res = score.score(records, m, split="all")
        g = res["hard_indef"]["recommended"][1]
        self.assertEqual(g["solvers"]["smf"]["total"], 1)
        self.assertEqual(g["solvers"]["smf"]["inaccurate"], 0)
        self.assertEqual(g["ratio"]["n"], 1)
        self.assertEqual(g["n_cases"], 1)
        self.assertEqual(set(g["report_only"]), {"sing"})
        self.assertNotIn("sing", g["times"]["smf"])
        self.assertIn("sing", score.render(res, "all"))

    def test_dropped_for_no_accurate_competitor_is_counted(self):
        m = manifest([case("a", "spd", "tuning"), case("b", "spd", "tuning")])
        records = [rec("a", "smf", 10.0), rec("a", "ma27", 20.0),
                   rec("b", "smf", 10.0), rec("b", "ma27", 20.0, eta=1e-3)]
        res = score.score(records, m, split="all")
        r = res["spd"]["recommended"][1]["ratio"]
        self.assertEqual((r["n"], r["no_accurate_competitor"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
