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
            rec("a", "ma57", 200.0),
            # faster but inaccurate: must not be "best"
            rec("a", "mkl_pardiso", 10.0, eta=1e-5),
            # smf itself is excluded from the competitor set even if it is fastest
        ]
        res = score.score(records, m, split="tuning", shift_ms=10.0)
        row = res["hard_indef"]["recommended"][1]["ratio"]
        self.assertEqual(row["n"], 1)
        self.assertTrue(math.isclose(row["geomean"], (100.0 + 10.0) / (200.0 + 10.0)))
        self.assertEqual(row["best_solver_counts"], {"ma57": 1})


    def test_implausible_throughput_is_not_a_competitor(self):
        # A "1-thread" record whose factor ran at 280 GF/s (rec(): factor_ms =
        # 0.6 * wall = 6 ms) cannot be one thread: it is neither solved nor
        # the best competitor, and it is listed.
        m = manifest([case("a", "hard_indef", "tuning")])
        fast = rec("a", "ssids", 10.0, flops=280e9 * 0.006)
        records = [rec("a", "smf", 100.0), rec("a", "ma57", 200.0), fast]
        res = score.score(records, m, split="tuning", shift_ms=10.0)
        g = res["hard_indef"]["recommended"][1]
        self.assertEqual(g["ratio"]["best_solver_counts"], {"ma57": 1})
        self.assertTrue(math.isclose(g["ratio"]["geomean"], 110.0 / 210.0))
        s = g["solvers"]["ssids"]
        self.assertEqual((s["total"], s["solved"], s["implausible"]), (1, 0, 1))
        self.assertEqual([(i["case"], i["solver"]) for i in g["implausible"]], [("a", "ssids")])
        self.assertTrue(math.isclose(g["implausible"][0]["gflops"], 280.0))
        md = score.render(res, "tuning")
        self.assertIn("Implausible", md)
        self.assertIn("| a | ssids |", md)

    def test_throughput_cap_scales_with_threads_and_can_be_disabled(self):
        m = manifest([case("a", "hard_indef", "tuning")])
        records = [rec("a", "smf", 100.0, threads=2), rec("a", "ma57", 200.0, threads=2),
                   rec("a", "ssids", 10.0, threads=2, flops=280e9 * 0.006)]  # 140 GF/s/thread
        res = score.score(records, m, split="tuning", shift_ms=10.0)
        self.assertEqual(res["hard_indef"]["recommended"][2]["ratio"]["best_solver_counts"],
                         {"ssids": 1})
        one = [rec("a", "smf", 100.0), rec("a", "ssids", 10.0, flops=280e9 * 0.006)]
        off = score.score(one, m, split="tuning", shift_ms=10.0, max_gflops_per_thread=0)
        self.assertEqual(off["hard_indef"]["recommended"][1]["solvers"]["ssids"]["implausible"], 0)

    def test_throughput_gate_uses_whole_factor_time_of_warm_sequences(self):
        # SSIDS pde_control_250: a 124 ms first step, then heavy steps; flops
        # (5.62e10) is a later step's count. First-step time would read 452
        # GF/s; over all steps it is 1.8 GF/s.
        r = rec("a", "ssids", 31730.0, mode="warm_sequence", batch_count=5,
                factor_ms=31211.0, first_factor_ms=124.1, refactor_ms=5012.1, flops=5.62e10)
        self.assertTrue(math.isclose(score.factor_gflops(r), 5.62e10 / 31211.0 / 1e6))
        self.assertEqual(score.record_status(r), "ok")

    def test_throughput_gate_skips_missing_or_bogus_flops(self):
        # MA97 reports zero or negative flops: no basis for a judgement.
        for flops in (None, 0, -1.02e9, float("nan")):
            r = rec("a", "ma97", 1.0, flops=flops)
            self.assertIsNone(score.factor_gflops(r), flops)
            self.assertEqual(score.record_status(r), "ok")

    def test_split_filter(self):
        m = manifest([case("a", "spd", "tuning"), case("b", "spd", "heldout")])
        records = [rec("a", "smf", 10.0), rec("a", "cholmod", 20.0),
                   rec("b", "smf", 40.0), rec("b", "cholmod", 20.0)]
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
        records = [rec("a", "smf", 100.0), rec("a", "ma57", 200.0),
                   rec("sing", "smf", 5.0, eta=1e-3), rec("sing", "ma57", 6.0)]
        res = score.score(records, m, split="all")
        g = res["hard_indef"]["recommended"][1]
        self.assertEqual(g["solvers"]["smf"]["total"], 1)
        self.assertEqual(g["solvers"]["smf"]["inaccurate"], 0)
        self.assertEqual(g["ratio"]["n"], 1)
        self.assertEqual(g["n_cases"], 1)
        self.assertEqual(set(g["report_only"]), {"sing"})
        self.assertNotIn("sing", g["times"]["smf"])
        self.assertIn("sing", score.render(res, "all"))

    # R43: fixed best-competitor set per class, independent of thread count.
    def test_ma27_and_cudss_are_reference_only(self):
        m = manifest([case("a", "small_kkt", "tuning"), case("b", "spd", "tuning")])
        records = [rec("a", "smf", 100.0), rec("a", "ma27", 10.0), rec("a", "cudss", 5.0),
                   rec("a", "ma57", 200.0),
                   rec("b", "smf", 100.0), rec("b", "ma27", 10.0), rec("b", "cudss", 5.0),
                   rec("b", "cholmod", 50.0)]
        res = score.score(records, m, split="all", shift_ms=10.0)
        r = res["small_kkt"]["recommended"][1]["ratio"]
        self.assertEqual(r["best_solver_counts"], {"ma57": 1})
        self.assertTrue(math.isclose(r["geomean"], 110.0 / 210.0))
        r = res["spd"]["recommended"][1]["ratio"]
        self.assertEqual(r["best_solver_counts"], {"cholmod": 1})
        # reference columns still listed
        self.assertIn("ma27", res["small_kkt"]["recommended"][1]["solvers"])

    def test_competitor_set_per_class(self):
        self.assertEqual(set(score.competitor_set("small_kkt")),
                         {"ma57", "ma97", "ssids", "mkl_pardiso", "mumps"})
        self.assertEqual(set(score.competitor_set("large_kkt")), set(score.competitor_set("hard_indef")))
        self.assertEqual(set(score.competitor_set("hard_indef")),
                         {"ma57", "ma97", "ssids", "mkl_pardiso", "mumps"})
        self.assertEqual(set(score.competitor_set("spd")),
                         {"cholmod", "mkl_pardiso", "ma57", "ma97", "ssids", "mumps"})
        # cholmod on an indefinite class is never best
        m = manifest([case("a", "hard_indef", "tuning")])
        res = score.score([rec("a", "smf", 100.0), rec("a", "cholmod", 1.0),
                           rec("a", "mumps", 300.0)], m, split="all", shift_ms=10.0)
        self.assertEqual(res["hard_indef"]["recommended"][1]["ratio"]["best_solver_counts"],
                         {"mumps": 1})

    def test_missing_competitor_reported_per_thread_table(self):
        m = manifest([case("a", "hard_indef", "tuning")])
        records = [rec("a", "smf", 100.0), rec("a", "ma57", 50.0), rec("a", "ma97", 80.0),
                   rec("a", "ma27", 10.0),
                   rec("a", "smf", 40.0, threads=8), rec("a", "ma97", 30.0, threads=8)]
        res = score.score(records, m, split="all", shift_ms=10.0)
        t1 = res["hard_indef"]["recommended"][1]["ratio"]
        t8 = res["hard_indef"]["recommended"][8]["ratio"]
        self.assertEqual(t1["competitor_set"], t8["competitor_set"])
        self.assertEqual(set(t1["missing_competitors"]), {"ssids", "mkl_pardiso", "mumps"})
        self.assertEqual(set(t8["missing_competitors"]),
                         {"ma57", "ssids", "mkl_pardiso", "mumps"})
        self.assertEqual(t1["best_solver_counts"], {"ma57": 1})
        self.assertEqual(t8["best_solver_counts"], {"ma97": 1})
        text = score.render(res, "all")
        self.assertIn("missing", text)
        self.assertIn("ma57", text.split("## hard_indef / recommended / 8")[1])

    def test_dropped_for_no_accurate_competitor_is_counted(self):
        m = manifest([case("a", "spd", "tuning"), case("b", "spd", "tuning")])
        records = [rec("a", "smf", 10.0), rec("a", "cholmod", 20.0),
                   rec("b", "smf", 10.0), rec("b", "cholmod", 20.0, eta=1e-3)]
        res = score.score(records, m, split="all")
        r = res["spd"]["recommended"][1]["ratio"]
        self.assertEqual((r["n"], r["no_accurate_competitor"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
