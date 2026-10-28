import pathlib
import unittest

import run_wsl


class WslPathTests(unittest.TestCase):
    def test_drive_paths_translate(self):
        self.assertEqual(run_wsl.to_wsl_path("C:\\Users\\daniel\\a b\\x.mtx"), "/mnt/c/Users/daniel/a b/x.mtx")
        self.assertEqual(run_wsl.to_wsl_path("D:/data/x.mtx"), "/mnt/d/data/x.mtx")

    def test_relative_resolves_against_cwd(self):
        cwd = pathlib.Path.cwd().resolve()
        drive = cwd.drive[0].lower()
        rest = cwd.as_posix().split(":/", 1)[1]
        self.assertEqual(run_wsl.to_wsl_path("benchmarks/x.mtx"), f"/mnt/{drive}/{rest}/benchmarks/x.mtx")

    def test_already_linux(self):
        self.assertEqual(run_wsl.to_wsl_path("/root/x"), "/root/x")


class WslCommandTests(unittest.TestCase):
    def test_command_shape(self):
        cmd = run_wsl.wsl_command(
            ["--case", "c1", "--mtx", "C:\\m\\a.mtx", "--JsonlOut", "C:\\o\\r.jsonl", "--threads", "1", "--warmup"],
            threads=1, driver="/root/ssids_bench/ssids_replay", distro="Ubuntu")
        self.assertEqual(cmd[:6], ["wsl.exe", "-d", "Ubuntu", "--exec", "/usr/bin/env", "OMP_NUM_THREADS=1"])
        self.assertIn("OMP_CANCELLATION=TRUE", cmd)
        self.assertIn("OMP_PROC_BIND=TRUE", cmd)
        self.assertIn("OPENBLAS_NUM_THREADS=1", cmd)
        args = cmd[cmd.index("/root/ssids_bench/ssids_replay") + 1:]
        self.assertEqual(args[args.index("--mtx") + 1], "/mnt/c/m/a.mtx")
        self.assertEqual(args[args.index("--JsonlOut") + 1], "/mnt/c/o/r.jsonl")
        self.assertEqual(args[args.index("--case") + 1], "c1")
        self.assertIn("--warmup", args)

    def test_solver_set(self):
        self.assertIn("ssids", run_wsl.WSL_SOLVERS)
        self.assertNotIn("smf", run_wsl.WSL_SOLVERS)


if __name__ == "__main__":
    unittest.main()


class ClassifyTests(unittest.TestCase):
    def test_driver_crash_is_failed_not_dead(self):
        outcome, why, dead = run_wsl.classify_no_record(139, "Segmentation fault")
        self.assertEqual(outcome, "Failed")
        self.assertFalse(dead)
        self.assertIn("139", why)

    def test_oom_kill_and_timeout_are_not_dead(self):
        self.assertEqual(run_wsl.classify_no_record(137, ""), ("Timeout", "ssids driver killed by timeout (rc=137)", False))
        self.assertFalse(run_wsl.classify_no_record(124, "")[2])

    def test_missing_wsl_or_driver_is_dead(self):
        for rc, text in ((0xFFFFFFFF, ""), (-1, ""), (127, ""),
                         (1, "There is no distribution with the supplied name."),
                         (1, "execvpe(/root/x) failed: No such file or directory")):
            outcome, _, dead = run_wsl.classify_no_record(rc, text)
            self.assertEqual(outcome, "Unavailable", (rc, text))
            self.assertTrue(dead, (rc, text))

    def test_command_has_timeout_guard(self):
        cmd = run_wsl.wsl_command(["--case", "c"], threads=1, driver="/d", distro="Ubuntu", timeout_s=57.9)
        i = cmd.index("/usr/bin/timeout")
        self.assertEqual(cmd[i:i + 5], ["/usr/bin/timeout", "-k", "5", "57", "/d"])


class RunDispatchTests(unittest.TestCase):
    """run.py --runner wsl: per-case driver failures do not disable ssids."""

    def _run(self, returncode):
        import json
        import subprocess
        import tempfile
        from unittest import mock

        import run

        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": [{"matrix_id": "c1", "mtx_path": "x.mtx"}]}))
            calls = []

            def fake(cmd, **kw):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, returncode, "", "boom")

            patches = (mock.patch.object(run.subprocess, "run", fake),
                       mock.patch.dict("os.environ", {"SMF_WSL_SSIDS_DRIVER": "/d"}))
            with patches[0], patches[1]:
                run.main(["--manifest", str(manifest), "--output", str(td / "o"), "--runner", "wsl",
                          "--solvers", "ssids", "--trials", "3", "--warmups", "0",
                          "--allow-short-trials", "--build-dir", str(td / "nobuild")])
            recs = [json.loads(l) for l in (td / "o" / "records.jsonl").read_text().splitlines()]
            return len(calls), recs

    def test_driver_crash_keeps_going(self):
        n, recs = self._run(139)
        self.assertEqual(n, 3)
        self.assertEqual({r["outcome"] for r in recs}, {"Failed"})

    def test_missing_wsl_stops_after_first(self):
        n, recs = self._run(0xFFFFFFFF)
        self.assertEqual(n, 1)
        self.assertEqual([r["outcome"] for r in recs], ["Unavailable"])
