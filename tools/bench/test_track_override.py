"""Tests for run.py / ab_run.py --track-override and the ipopt-plugin track in
score.py / ab_compare.py."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import ab_compare  # noqa: E402
import ab_run  # noqa: E402
import run  # noqa: E402
import score  # noqa: E402

CASES = [
    {"matrix_id": "a", "class": "hard_indef", "split": "tuning", "mode": "cold",
     "track": "recommended", "mtx_path": "a.mtx"},
    {"matrix_id": "a", "class": "hard_indef", "split": "tuning", "mode": "warm_sequence",
     "track": "ipopt", "mtx_path": "a.mtx", "seq_values_path": "a.vals", "ordering": "matching"},
    {"matrix_id": "b", "class": "spd", "split": "heldout", "mode": "cold",
     "track": "recommended", "mtx_path": "b.mtx"},
]


def rec(matrix, wall, track="ipopt-plugin", solver="smf", trial=0):
    return {"matrix_id": matrix, "solver_id": solver, "track": track, "threads": 1,
            "warmup": False, "outcome": "Success", "mode": "cold", "trial_id": trial,
            "wall_ms": wall, "analyse_ms": wall * 0.5, "factor_ms": wall * 0.4,
            "solve_ms": wall * 0.1, "refine_ms": 0.0, "peak_rss_bytes": 1000,
            "validation": {"finite": True, "eta_inf": 1e-15, "omega": 1e-14},
            "inertia": {"pos": 6, "neg": 4, "zero": 0}, "n": 10}


class TrackOverrideCasesTests(unittest.TestCase):
    def test_one_entry_per_matrix_preferring_ipopt(self):
        out = score.track_override_cases(CASES, "ipopt-plugin")
        self.assertEqual([c["matrix_id"] for c in out], ["a", "b"])
        self.assertEqual({c["track"] for c in out}, {"ipopt-plugin"})
        a = out[0]
        self.assertEqual(a["mode"], "warm_sequence")
        self.assertEqual(a["seq_values_path"], "a.vals")
        self.assertEqual(a["manifest_track"], "ipopt")
        self.assertNotIn("ordering", a)
        self.assertEqual(out[1]["manifest_track"], "recommended")
        # the manifest entries themselves are untouched
        self.assertEqual(CASES[1]["track"], "ipopt")

    def test_ipopt_entry_listed_first_is_kept(self):
        out = score.track_override_cases(list(reversed(CASES[:2])), "ipopt-plugin")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["manifest_track"], "ipopt")

    def test_case_index_covers_override_tracks(self):
        idx = score.manifest_case_index(CASES)
        self.assertIn(("a", "ipopt-plugin"), idx)
        self.assertEqual(idx[("a", "ipopt-plugin")]["class"], "hard_indef")
        self.assertIn(("a", "recommended"), idx)
        self.assertIn(("b", "ipopt-plugin"), idx)


class RunTrackOverrideTests(unittest.TestCase):
    def _run(self, extra):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"solvers": ["smf", "ma57"], "cases": CASES}))
            binary = td / "bench_replay.exe"
            binary.write_bytes(b"")
            calls = []
            self.envs = []

            def fake(cmd, **kw):
                calls.append(cmd)
                self.envs.append(kw.get("env"))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(run.subprocess, "run", fake):
                rc = run.main(["--manifest", str(manifest), "--output", str(td / "o"),
                               "--binary", str(binary), "--trials", "1", "--warmups", "0",
                               "--allow-short-trials"] + extra)
            return rc, calls

    @staticmethod
    def _arg(cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def test_ipopt_plugin_runs_each_case_once_smf_only(self):
        rc, calls = self._run(["--track-override", "ipopt-plugin"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual({self._arg(c, "--solver") for c in calls}, {"smf"})
        self.assertEqual({self._arg(c, "--track") for c in calls}, {"ipopt-plugin"})
        by_case = {self._arg(c, "--case"): c for c in calls}
        self.assertEqual(self._arg(by_case["a"], "--mode"), "warm_sequence")
        self.assertEqual(self._arg(by_case["a"], "--seq-values"), "a.vals")
        self.assertNotIn("--ordering", by_case["a"])  # per-case pins dropped on override tracks
        self.assertEqual(self._arg(by_case["b"], "--mode"), "cold")

    def test_case_filter_applies(self):
        _, calls = self._run(["--track-override", "ipopt-plugin", "--cases", "a"])
        self.assertEqual([self._arg(c, "--case") for c in calls], ["a"])

    def test_without_override_every_entry_runs(self):
        _, calls = self._run(["--solvers", "smf"])
        self.assertEqual(sorted(self._arg(c, "--track") for c in calls),
                         ["ipopt", "recommended", "recommended"])

    def test_only_override_tracks_accepted(self):
        for bad in ("bogus", "recommended", "ipopt"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self._run(["--track-override", bad])

    def test_no_smf_selected_is_an_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc, calls = self._run(["--track-override", "ipopt-plugin", "--solvers", "ma57"])
        self.assertEqual(rc, 2)
        self.assertEqual(calls, [])
        self.assertIn("smf only", err.getvalue())

    def test_threads_follow_omp_num_threads(self):
        # The plugin's thread count comes from OMP_NUM_THREADS, which every
        # run exports as its --threads value: no warning, N-thread rows run N.
        with mock.patch.dict(run.os.environ, {}, clear=False):
            run.os.environ.pop("SMF_NUM_THREADS", None)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                _, calls = self._run(["--track-override", "ipopt-plugin", "--threads", "1,8"])
        self.assertNotIn("warning", err.getvalue())
        self.assertEqual(sorted({(self._arg(c, "--threads"), e["OMP_NUM_THREADS"])
                                 for c, e in zip(calls, self.envs)}),
                         [("1", "1"), ("8", "8")])
        self.assertTrue(all("SMF_NUM_THREADS" not in e for e in self.envs))

    def test_smf_num_threads_override_warns(self):
        with mock.patch.dict(run.os.environ, {"SMF_NUM_THREADS": "4"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                _, calls = self._run(["--track-override", "ipopt-plugin", "--threads", "1"])
        self.assertIn("SMF_NUM_THREADS=4 overrides --threads", err.getvalue())
        self.assertTrue(all(e["SMF_NUM_THREADS"] == "4" for e in self.envs))


class AbRunPassThroughTests(unittest.TestCase):
    def test_track_override_passed_to_run(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": CASES}))
            calls = []

            def fake(cmd, **kw):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch.object(ab_run.subprocess, "run", fake):
                ab_run.main(["--manifest", str(manifest), "--output", str(td / "o"),
                             "--variant", "A=X=1", "--variant", "B=X=2", "--rounds", "1",
                             "--cases", "a", "--track-override", "ipopt-plugin"])
            self.assertEqual(len(calls), 2)
            for cmd in calls:
                self.assertEqual(cmd[cmd.index("--track-override") + 1], "ipopt-plugin")

            calls.clear()
            with mock.patch.object(ab_run.subprocess, "run", fake):
                ab_run.main(["--manifest", str(manifest), "--output", str(td / "o2"),
                             "--variant", "A=X=1", "--variant", "B=X=2", "--rounds", "1",
                             "--cases", "a"])
            self.assertTrue(all("--track-override" not in cmd for cmd in calls))

    def test_per_variant_binary(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": CASES}))
            calls = []

            def fake(cmd, **kw):
                calls.append((cmd, kw["env"]))
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch.object(ab_run.subprocess, "run", fake):
                ab_run.main(["--manifest", str(manifest), "--output", str(td / "o"),
                             "--binary", "default.exe",
                             "--variant", "A=SMF_AB_BINARY=a.exe,X=1", "--variant", "B=X=2",
                             "--rounds", "2", "--cases", "a"])
            self.assertEqual(len(calls), 4)
            for cmd, env in calls:
                binary = cmd[cmd.index("--binary") + 1]
                self.assertEqual(binary, "a.exe" if env["X"] == "1" else "default.exe")
                self.assertNotIn("SMF_AB_BINARY", env)

    def test_ab_run_rejects_other_tracks(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            ab_run.main(["--manifest", "m.json", "--output", "o", "--variant", "A=X=1",
                         "--variant", "B=X=2", "--track-override", "ipopt"])


class ScoreAndCompareTests(unittest.TestCase):
    def test_score_reports_ipopt_plugin_track(self):
        records = [rec("a", 100.0), rec("b", 50.0)]
        res = score.score(records, {"cases": CASES}, split="all")
        self.assertIn("ipopt-plugin", res["hard_indef"])
        self.assertIn("ipopt-plugin", res["spd"])
        g = res["hard_indef"]["ipopt-plugin"][1]
        self.assertEqual(g["solvers"]["smf"]["solved"], 1)
        text = score.render(res, "all")
        self.assertIn("hard_indef / ipopt-plugin / 1 thread(s)", text)
        self.assertIn("Track ipopt-plugin runs smf only", text)
        self.assertNotIn("missing in this table: ma57", text)

    def test_ab_compare_self_vs_self(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            (td / "m.json").write_text(json.dumps({"cases": CASES}))
            for side, scale in (("A", 1.0), ("B", 0.5)):
                (td / side).mkdir()
                with (td / side / "records.jsonl").open("w") as f:
                    for r in (rec("a", 100.0 * scale), rec("b", 40.0 * scale)):
                        f.write(json.dumps(r) + "\n")
            out = td / "cmp.md"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = ab_compare.main(["--a", str(td / "A"), "--b", str(td / "B"),
                                     "--manifest", str(td / "m.json"), "--output", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("| a | plugin | 1 | hard_indef |", text)
            self.assertIn("| b | plugin | 1 | spd |", text)
            self.assertIn("| 1 | all | tuning | 1 |", text)


if __name__ == "__main__":
    unittest.main()


class AbSeqStepsTests(unittest.TestCase):
    def test_seq_steps_passed_to_run(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": CASES}))
            calls = []

            def fake(cmd, **kw):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch.object(ab_run.subprocess, "run", fake):
                ab_run.main(["--manifest", str(manifest), "--output", str(td / "o"),
                             "--variant", "A=X=1", "--variant", "B=X=2", "--rounds", "1",
                             "--cases", "a", "--seq-steps", "all"])
            self.assertEqual(len(calls), 2)
            for cmd in calls:
                self.assertEqual(cmd[cmd.index("--seq-steps") + 1], "all")

    def test_refactor_metric_summarized(self):
        r = rec("a", 10.0)
        r["refactor_ms"] = 2.5
        self.assertEqual(ab_compare.summarize([r])["refactor"], 2.5)
