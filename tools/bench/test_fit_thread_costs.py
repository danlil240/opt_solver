"""Tests for tools/bench/fit_thread_costs.py on synthetic traces and records."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import fit_thread_costs as fit  # noqa: E402

# Case A: 10 split top fronts (tile tasks). The model with no per-split cost
# picks 8 threads (4 at max 4); measured, 1 thread is fastest.
TRACE_A = """\
analyse: something unrelated
plan-factor T=1 ok=1 sub_total=0 max_subtree=0 top_total=0 cp_top=0 top_fronts=0 split_top_fronts=0 split_top_work=0 total=1e+08
plan-factor T=2 ok=1 sub_total=6e+07 max_subtree=5e+06 top_total=4e+07 cp_top=3e+07 top_fronts=12 split_top_fronts=10 split_top_work=3.5e+07 total=1e+08
plan-factor T=4 ok=1 sub_total=6e+07 max_subtree=5e+06 top_total=4e+07 cp_top=1.6e+07 top_fronts=12 split_top_fronts=10 split_top_work=3.5e+07 total=1e+08
plan-factor T=8 ok=1 sub_total=6e+07 max_subtree=5e+06 top_total=4e+07 cp_top=1.2e+07 top_fronts=12 split_top_fronts=10 split_top_work=3.5e+07 total=1e+08
plan-factor T=16 ok=0 sub_total=1e+08 max_subtree=1e+08 top_total=0 cp_top=0 top_fronts=0 split_top_fronts=0 split_top_work=0 total=1e+08
plan-factor chosen=8
plan-solve T=1 sub_total=1e+06 max_subtree=100000 top_total=100000 top_cp=50000 n_top=10 top_rows=1000
plan-solve T=2 sub_total=1e+06 max_subtree=100000 top_total=100000 top_cp=50000 n_top=10 top_rows=1000
plan-solve T=4 sub_total=1e+06 max_subtree=100000 top_total=100000 top_cp=50000 n_top=10 top_rows=1000
plan-solve T=8 sub_total=1e+06 max_subtree=100000 top_total=100000 top_cp=50000 n_top=10 top_rows=1000
plan-solve T=16 sub_total=1e+06 max_subtree=100000 top_total=100000 top_cp=50000 n_top=10 top_rows=1000
plan-solve chosen=8
step 0 (stored 0): static=0 delays=0 rescaled=0 inertia=1/1/0 factor_ms=1.000 threads=8 subtrees=9 top=12 pops=12 sub_ms=0.1 top_ms=0.1 gather_ms=0.1 solve_threads=8
plan-factor T=1 ok=1 sub_total=0 max_subtree=0 top_total=0 cp_top=0 top_fronts=0 split_top_fronts=0 split_top_work=0 total=7
plan-factor chosen=1
plan-solve T=1 sub_total=7 max_subtree=7 top_total=7 top_cp=7 n_top=1 top_rows=1
plan-solve chosen=1
"""

# Case B: no split fronts, a balanced tree that scales to 16 threads.
B_LINE = ("plan-factor T={T} ok=1 sub_total=9e+08 max_subtree=1e+07 top_total=1e+08 "
          "cp_top=1e+07 top_fronts=20 split_top_fronts=0 split_top_work=0 total=1e+09")
TRACE_B = "\n".join(
    ["plan-factor T=1 ok=1 sub_total=0 max_subtree=0 top_total=0 cp_top=0 top_fronts=0 "
     "split_top_fronts=0 split_top_work=0 total=1e+09"]
    + [B_LINE.format(T=T) for T in (2, 4, 8, 16)]
    + ["plan-factor chosen=16", "step 0 (stored 0): factor_ms=1.0"]) + "\n"

FACTOR_A = {1: 10.0, 2: 12.0, 4: 14.0, 8: 15.0, 16: 16.0}
FACTOR_B = {1: 100.0, 2: 55.0, 4: 30.0, 8: 18.0, 16: 12.0}


def case(name, trace, factor, solve=None):
    return {"key": (name, "recommended"), "class": "hard_indef", "trace": fit.parse_trace(trace),
            "factor": factor, "solve": solve or {}}


class ParseTraceTest(unittest.TestCase):
    def test_first_block_only(self):
        tr = fit.parse_trace(TRACE_A)
        self.assertEqual(sorted(tr["factor"]), [1, 2, 4, 8, 16])
        self.assertEqual(tr["factor_chosen"], 8)
        k2 = tr["factor"][2]
        self.assertTrue(k2["ok"])
        self.assertEqual(k2["sub_total"], 6e7)
        self.assertEqual(k2["cp_top"], 3e7)
        self.assertEqual(k2["top_fronts"], 12)
        self.assertEqual(k2["split_top_fronts"], 10)
        self.assertEqual(k2["split_top_work"], 3.5e7)
        self.assertEqual(k2["total"], 1e8)
        self.assertFalse(tr["factor"][16]["ok"])
        self.assertEqual(tr["factor"][1]["total"], 1e8)  # not the later block's 7
        self.assertEqual(sorted(tr["solve"]), [1, 2, 4, 8, 16])
        self.assertEqual(tr["solve_chosen"], 8)
        s = tr["solve"][4]
        self.assertEqual((s["sub_total"], s["max_subtree"], s["top_total"], s["top_cp"]),
                         (1e6, 1e5, 1e5, 5e4))
        self.assertEqual((s["n_top"], s["top_rows"]), (10, 1000))

    def test_missing_blocks(self):
        tr = fit.parse_trace("step 0 (stored 0): factor_ms=1.0\n")
        self.assertEqual(tr["factor"], {})
        self.assertIsNone(tr["factor_chosen"])
        self.assertEqual(tr["solve"], {})
        self.assertIsNone(tr["solve_chosen"])


class ModelTest(unittest.TestCase):
    def test_candidates_and_tie_rule(self):
        self.assertEqual(fit.candidate_threads(1), [1])
        self.assertEqual(fit.candidate_threads(3), [1, 2, 3])
        self.assertEqual(fit.candidate_threads(12), [1, 2, 4, 8, 12])
        self.assertEqual(fit.candidate_threads(16), [1, 2, 4, 8, 16])
        # Ascending: the first within 2% of the least.
        self.assertEqual(fit.least_cost_threads([(1, 100.0), (2, 99.0), (4, 98.0)]), 2)
        self.assertEqual(fit.least_cost_threads([(1, 100.0), (2, 101.0), (4, 98.5)]), 1)
        # cost < 0: no candidate.
        self.assertEqual(fit.least_cost_threads([(1, 100.0), (2, -1.0), (4, 50.0)]), 4)

    def test_factor_hand_computed(self):
        tr = fit.parse_trace(TRACE_A)["factor"]
        c = fit.FactorConsts(1e5, 5e3, 0.0)
        # cost(2) = max(5e6, 3e7) + max(3e7, 2e7) + 2e5 + 12 * 5e3
        self.assertAlmostEqual(fit.factor_cost(tr[2], c), 6.026e7)
        self.assertAlmostEqual(fit.factor_cost(tr[4], c), 3.146e7)
        self.assertAlmostEqual(fit.factor_cost(tr[8], c), 2.036e7)
        self.assertEqual(fit.factor_cost(tr[1], c), 1e8)
        self.assertLess(fit.factor_cost(tr[16], c), 0.0)
        self.assertEqual(fit.pick_factor_threads(tr, 16, c), 8)
        self.assertEqual(fit.pick_factor_threads(tr, 8, c), 8)
        self.assertEqual(fit.pick_factor_threads(tr, 4, c), 4)
        # 10 split fronts at 1e7 each: every parallel candidate costs > 1e8.
        split = fit.FactorConsts(1e5, 5e3, 1e7)
        self.assertAlmostEqual(fit.factor_cost(tr[8], split), 1.2036e8)
        self.assertEqual(fit.pick_factor_threads(tr, 16, split), 1)
        self.assertEqual(fit.pick_factor_threads(tr, 4, split), 1)

    def test_factor_light_tree(self):
        tr = fit.parse_trace("plan-factor T=1 ok=1 sub_total=0 max_subtree=0 top_total=0 cp_top=0 "
                             "top_fronts=0 split_top_fronts=0 split_top_work=0 total=5e+06\n"
                             "plan-factor chosen=1\n")["factor"]
        self.assertEqual(fit.pick_factor_threads(tr, 16, fit.FactorConsts(1e5, 5e3, 0.0)), 1)

    def test_solve_hand_computed(self):
        tr = fit.parse_trace(TRACE_A)["solve"]
        c = fit.SolveConsts(2e4, 2e3, 2.0)
        self.assertEqual(fit.solve_cost(tr[1], 1, c), 1.1e6)
        # max(1e5, 1e6 / T) + 5e4 + T 2e4 + 10 * 2e3 + T * 1000 * 2
        self.assertAlmostEqual(fit.solve_cost(tr[2], 2, c), 6.14e5)
        self.assertAlmostEqual(fit.solve_cost(tr[8], 8, c), 3.71e5)
        self.assertAlmostEqual(fit.solve_cost(tr[16], 16, c), 5.22e5)
        self.assertEqual(fit.pick_solve_threads(tr, 16, c), 8)
        self.assertEqual(fit.pick_solve_threads(tr, 4, c), 4)
        # A serial solve layout (no plan, or only T = 1 traced) stays at 1.
        self.assertEqual(fit.pick_solve_threads({}, 16, c), 1)
        self.assertEqual(fit.pick_solve_threads({1: tr[1]}, 16, c), 1)


class LossTest(unittest.TestCase):
    def test_zero_at_measured_best(self):
        self.assertEqual(fit.case_loss(16, FACTOR_B, 16), 0.0)
        self.assertEqual(fit.case_loss(8, FACTOR_B, 8), 0.0)  # best T <= 8
        self.assertAlmostEqual(fit.case_loss(8, FACTOR_B, 16), math.log(18.0 / 12.0))
        self.assertIsNone(fit.case_loss(2, {1: 5.0}, 16))  # not measured at 2

    def test_evaluate_skips_fast_cases(self):
        cases = [case("a", TRACE_A, FACTOR_A), case("b", TRACE_B, FACTOR_B),
                 case("tiny", TRACE_B, {1: 0.5, 16: 0.4})]
        good = fit.evaluate(cases, "factor", fit.FactorConsts(1e3, 5e3, 1e7))
        self.assertEqual(good["loss"], 0.0)
        self.assertEqual(good["used"], 2)
        self.assertEqual(good["skipped"], 1)
        self.assertEqual(good["a1"], 0)
        self.assertEqual(good["a2"], 0)
        bad = fit.evaluate(cases, "factor", fit.FactorConsts(1e3, 5e3, 0.0))
        # a picks 8 at max 16 and 8: 15 / 10 twice; 15 > 1.05 x 10 (A1).
        self.assertAlmostEqual(bad["loss"], 2 * math.log(1.5))
        self.assertEqual(bad["a1"], 1)
        self.assertEqual(bad["a2"], 0)
        # A case without a measurement at some candidate T counts at no grid
        # point (a point choosing an unmeasured T is not rewarded).
        gap = cases + [case("gap", TRACE_B, {1: 100.0, 4: 30.0, 8: 18.0, 16: 12.0})]
        for c in (fit.FactorConsts(1e3, 5e3, 1e7), fit.FactorConsts(1e7, 5e3, 0.0)):
            ev = fit.evaluate(gap, "factor", c)
            self.assertEqual((ev["used"], ev["skipped_fast"], ev["skipped_incomplete"]), (2, 1, 1))
        row_a = [r for r in bad["rows"] if r["case"] == "a"][0]
        self.assertEqual(row_a["chosen"], {16: 8, 8: 8})
        self.assertEqual(row_a["best"], 1)
        self.assertTrue(row_a.get("a1"))
        self.assertAlmostEqual(row_a["ratio"], 1.5)


class GridTest(unittest.TestCase):
    def test_known_optimum(self):
        cases = [case("a", TRACE_A, FACTOR_A), case("b", TRACE_B, FACTOR_B)]
        grid = [fit.FactorConsts(pt, ptf, ps)
                for pt in (1e3, 1e7) for ptf in (5e3,) for ps in (0.0, 1e7)]
        ranked = fit.grid_search(cases, "factor", grid)
        self.assertEqual(ranked[0][0], fit.FactorConsts(1e3, 5e3, 1e7))
        self.assertEqual(ranked[0][1]["loss"], 0.0)
        self.assertTrue(all(r[1]["loss"] > 0.0 for r in ranked[1:]))

    def test_default_grids(self):
        self.assertEqual(len(fit.FACTOR_GRID), 5 * 5 * 6)
        self.assertEqual(len(fit.SOLVE_GRID), 4 * 4 * 4)
        self.assertIn(fit.FactorConsts(1e5, 5e3, 0.0), fit.FACTOR_GRID)
        self.assertIn(fit.SolveConsts(2e4, 2e3, 2.0), fit.SOLVE_GRID)


def rec(matrix, threads, factor_ms, solve_ms, track="recommended", solver="smf"):
    return {
        "matrix_id": matrix, "solver_id": solver, "track": track, "threads": threads,
        "warmup": False, "outcome": "Success", "mode": "cold", "wall_ms": 100.0,
        "analyse_ms": 1.0, "factor_ms": factor_ms, "solve_ms": solve_ms, "refine_ms": 0.0,
        "validation": {"finite": True, "eta_inf": 1e-15, "omega": 1e-14},
        "inertia": {"pos": 6, "neg": 4, "zero": 0}, "n": 10,
    }


class EndToEndTest(unittest.TestCase):
    def test_main_writes_report(self):
        manifest = {"cases": [
            {"matrix_id": "a", "track": "recommended", "class": "hard_indef"},
            {"matrix_id": "b", "track": "recommended", "class": "large_kkt"},
            {"matrix_id": "notrace", "track": "recommended", "class": "spd"},
        ]}
        records = []
        for t in (1, 2, 4, 8, 16):
            records.append(rec("a", t, FACTOR_A[t], 5.0 - 0.1 * t))
            records.append(rec("b", t, FACTOR_B[t], 2.0))
            records.append(rec("notrace", t, 10.0, 2.0))
            records.append(rec("a", t, 0.001, 0.001, solver="ma57"))  # not smf: ignored
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "truth").mkdir()
            (root / "truth" / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            (root / "traces").mkdir()
            (root / "traces" / "a__recommended.txt").write_text(TRACE_A, encoding="utf-8")
            (root / "traces" / "b__recommended.txt").write_text(TRACE_B, encoding="utf-8")
            (root / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = root / "fit.md"
            rc = fit.main(["--truth", str(root / "truth"), "--traces", str(root / "traces"),
                           "--manifest", str(root / "m.json"), "--output", str(out)])
            self.assertEqual(rc, 0)
            md = out.read_text(encoding="utf-8")
        self.assertIn("## Factor: 5 best", md)
        self.assertIn("## Solve: 5 best", md)
        self.assertIn("notrace", md)  # listed as missing a trace
        self.assertIn("| a | recommended |", md)


if __name__ == "__main__":
    unittest.main()
