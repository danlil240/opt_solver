"""Tests for tools/bench/thread_scaling.py on synthetic records."""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import thread_scaling  # noqa: E402


def rec(matrix, threads, factor_ms, solve_ms, *, solver="smf", track="recommended", **extra):
    r = {
        "matrix_id": matrix, "solver_id": solver, "track": track,
        "threads": threads, "warmup": False, "outcome": "Success",
        "mode": "cold", "wall_ms": 100.0,
        "analyse_ms": 20.0, "factor_ms": factor_ms,
        "solve_ms": solve_ms, "refine_ms": 0.0,
        "validation": {"finite": True, "eta_inf": 1e-15, "omega": 1e-14},
        "inertia": {"pos": 6, "neg": 4, "zero": 0},
        "n": 10,
    }
    r.update(extra)
    return r


MANIFEST = {"cases": [
    {"matrix_id": "slow_par", "track": "recommended", "class": "large_kkt", "split": "tuning"},
    {"matrix_id": "scales", "track": "recommended", "class": "spd", "split": "tuning"},
]}


def records():
    out = []
    for t, f, s in ((1, 10.0, 5.0), (8, 30.0, 4.0), (16, 50.0, 4.0)):
        out.append(rec("slow_par", t, f, s))
    for t, f, s in ((1, 10.0, 5.0), (8, 6.0, 3.0), (16, 4.0, 2.0)):
        out.append(rec("scales", t, f, s))
    # Other solvers and warmups are ignored.
    out.append(rec("slow_par", 8, 1.0, 1.0, solver="ma57"))
    out.append(rec("slow_par", 8, 999.0, 999.0, warmup=True))
    return out


def row_for(md, case):
    rows = [ln for ln in md.splitlines() if ln.startswith("|") and f"| {case} |" in ln]
    assert len(rows) == 1, rows
    return rows[0]


class ThreadScalingTest(unittest.TestCase):
    def test_flags_and_ratios(self):
        md = thread_scaling.report(records(), MANIFEST, min_ms=1.0)
        row = row_for(md, "slow_par")
        self.assertIn("8:3.00", row)
        self.assertIn("16:5.00", row)
        self.assertIn("FAC>1T", row)
        self.assertIn("16>8", row)
        self.assertNotIn("SOL>1T", row)
        self.assertIn("| 1 |", row)  # best thread count

    def test_scaling_case_has_no_flags(self):
        md = thread_scaling.report(records(), MANIFEST, min_ms=1.0)
        row = row_for(md, "scales")
        for flag in ("FAC>1T", "SOL>1T", "16>8"):
            self.assertNotIn(flag, row)
        self.assertIn("| 16 |", row)

    def test_summary_line(self):
        md = thread_scaling.report(records(), MANIFEST, min_ms=1.0)
        self.assertIn("FAC>1T: 1, SOL>1T: 0, 16>8: 1", md)

    def test_min_ms_suppresses_flags_on_tiny_cases(self):
        md = thread_scaling.report(records(), MANIFEST, min_ms=100.0)
        self.assertIn("FAC>1T: 0, SOL>1T: 0, 16>8: 0", md)

    def test_main_reads_nested_record_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "run" / "sub").mkdir(parents=True)
            (root / "run" / "sub" / "records.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records()) + "\n", encoding="utf-8")
            man = root / "m.json"
            man.write_text(json.dumps(MANIFEST), encoding="utf-8")
            out = root / "out.md"
            rc = thread_scaling.main(["--records", str(root / "run"), "--manifest", str(man),
                                      "--output", str(out)])
            self.assertEqual(rc, 0)
            self.assertIn("FAC>1T: 1, SOL>1T: 0, 16>8: 1", out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
