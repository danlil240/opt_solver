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

    def test_thread_env_caps_openmp_and_keeps_blas_serial(self):
        # SSIDS ignores OMP_NUM_THREADS (hwloc sizes its regions): the thread
        # limit is what caps it, and BLAS threads would oversubscribe.
        cmd = run_wsl.wsl_command(["--case", "c"], threads=8, driver="/d", distro="Ubuntu")
        env = cmd[cmd.index("/usr/bin/env") + 1:cmd.index("/d")]
        self.assertIn("OMP_NUM_THREADS=8", env)
        self.assertIn("OMP_THREAD_LIMIT=8", env)
        self.assertIn("OPENBLAS_NUM_THREADS=1", env)
        self.assertIn("MKL_NUM_THREADS=1", env)
        self.assertNotIn("OPENBLAS_NUM_THREADS=8", env)

    def test_solver_set(self):
        self.assertIn("ssids", run_wsl.WSL_SOLVERS)
        self.assertNotIn("smf", run_wsl.WSL_SOLVERS)


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


class WslSmfCommandTests(unittest.TestCase):
    def test_wsl_smf_command_env_and_paths(self):
        cmd = run_wsl.wsl_smf_command(
            ["--solver", "smf", "--mtx", r"C:\data\a.mtx", "--JsonlOut", r"C:\out\r.jsonl"],
            threads=8,
            binary="/home/u/build/bench_replay",
            distro="Ubuntu",
            timeout_s=300,
        )
        self.assertEqual(cmd[:5], ["wsl.exe", "-d", "Ubuntu", "--exec", "/usr/bin/env"])
        bin_idx = cmd.index("/home/u/build/bench_replay")
        env = cmd[cmd.index("/usr/bin/env") + 1:bin_idx]
        self.assertIn("OMP_NUM_THREADS=8", env)
        self.assertIn("OPENBLAS_NUM_THREADS=1", env)
        self.assertIn("MKL_NUM_THREADS=1", env)
        self.assertNotIn("OMP_CANCELLATION=TRUE", env)
        self.assertNotIn("OMP_PROC_BIND=TRUE", env)
        timeout_idx = cmd.index("/usr/bin/timeout")
        self.assertEqual(cmd[timeout_idx:timeout_idx + 4], ["/usr/bin/timeout", "-k", "5", "300"])
        self.assertEqual(cmd[timeout_idx + 4], "/home/u/build/bench_replay")
        args = cmd[bin_idx + 1:]
        self.assertEqual(args[args.index("--mtx") + 1], "/mnt/c/data/a.mtx")
        self.assertEqual(args[args.index("--JsonlOut") + 1], "/mnt/c/out/r.jsonl")

    def test_smf_driver_path_env_override(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"SMF_WSL_SMF_BINARY": "/custom/bench_replay"}):
            self.assertEqual(run_wsl.smf_driver_path(), "/custom/bench_replay")

        clean_env = {k: v for k, v in os.environ.items() if k != "SMF_WSL_SMF_BINARY"}
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with mock.patch.object(run_wsl, "wsl_home", return_value="/home/alice"):
                self.assertEqual(
                    run_wsl.smf_driver_path(),
                    "/home/alice/opt_solver_wsl/build/benchmarks/bench_replay",
                )
            with mock.patch.object(run_wsl, "wsl_home", return_value=None):
                self.assertEqual(
                    run_wsl.smf_driver_path(),
                    "/root/opt_solver_wsl/build/benchmarks/bench_replay",
                )


class InventoryAndPanuaEnvTests(unittest.TestCase):
    def test_inventory_environment_includes_panua_and_gpu_keys(self):
        import os
        import subprocess
        import tempfile
        from unittest import mock

        import run

        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("SMF_PANUA_PARDISO_LIB", "PANUA_LIC_PATH")}
        with tempfile.TemporaryDirectory() as td:
            fake_home = pathlib.Path(td)
            with mock.patch.dict(os.environ, clean_env, clear=True), \
                 mock.patch.object(pathlib.Path, "home", return_value=fake_home), \
                 mock.patch.object(run.subprocess, "Popen", side_effect=FileNotFoundError("no nvidia-smi")):
                info = run.inventory_environment(pathlib.Path("build"))
                for key in ("panua_pardiso_lib", "panua_lic_path", "panua_license_check", "gpu"):
                    self.assertIn(key, info)
                    self.assertIsNone(info[key])

            # With Panua lib, check_license, panua.lic in home, and nvidia-smi present
            panua_root = fake_home / "panua"
            (panua_root / "lib").mkdir(parents=True)
            (panua_root / "bin").mkdir(parents=True)
            lib_file = panua_root / "lib" / "libpardiso800.dll"
            lib_file.write_text("", encoding="utf-8")
            chk_file = panua_root / "bin" / "check_license.exe"
            chk_file.write_text("", encoding="utf-8")
            (fake_home / "panua.lic").write_text("LIC", encoding="utf-8")

            def fake_popen(cmd, **kw):
                proc = mock.MagicMock()
                proc.returncode = 0
                if "nvidia-smi" in cmd[0]:
                    proc.communicate.return_value = ("NVIDIA RTX 4090, 550.0, 24564 MiB\n", "")
                else:
                    proc.communicate.return_value = ("License valid until 2027-01-01\n", "")
                return proc

            with mock.patch.dict(os.environ, {"SMF_PANUA_PARDISO_LIB": str(lib_file)}), \
                 mock.patch.object(pathlib.Path, "home", return_value=fake_home), \
                 mock.patch.object(run.subprocess, "Popen", side_effect=fake_popen):
                info = run.inventory_environment(pathlib.Path("build"))
                self.assertEqual(info["panua_pardiso_lib"], str(lib_file))
                self.assertEqual(info["panua_lic_path"], str(fake_home / "panua.lic"))
                self.assertEqual(info["panua_license_check"], "License valid until 2027-01-01")
                self.assertEqual(info["gpu"], "NVIDIA RTX 4090, 550.0, 24564 MiB")

    def test_panua_pardiso_sets_pardisolicmessage_env(self):
        import json
        import os
        import subprocess
        import tempfile
        from unittest import mock

        import run

        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": [{"matrix_id": "c1", "mtx_path": "x.mtx"}]}))
            fake_bin = td / "bench_replay.exe"
            fake_bin.write_text("", encoding="utf-8")
            captured_envs = []

            def fake_run(cmd, **kw):
                captured_envs.append(kw.get("env", {}))
                return subprocess.CompletedProcess(cmd, 0, "", "")

            clean_env = {k: v for k, v in os.environ.items() if k != "PARDISOLICMESSAGE"}
            with mock.patch.dict(os.environ, clean_env, clear=True), \
                 mock.patch.object(run.subprocess, "run", side_effect=fake_run):
                run.main([
                    "--manifest", str(manifest),
                    "--output", str(td / "out"),
                    "--solvers", "panua_pardiso",
                    "--trials", "1",
                    "--warmups", "0",
                    "--allow-short-trials",
                    "--binary", str(fake_bin),
                ])
            self.assertEqual(len(captured_envs), 1)
            self.assertEqual(captured_envs[0].get("PARDISOLICMESSAGE"), "1")


class RunWslSolverFlagTests(unittest.TestCase):
    def test_run_wsl_main_accepts_solver_smf_and_ssids(self):
        from unittest import mock

        import run

        recorded_cmds = []

        def fake_run_main(cmd):
            recorded_cmds.append(cmd)
            return 0

        with mock.patch.object(run, "main", side_effect=fake_run_main):
            rc_smf = run_wsl.main(["--solver", "smf", "--manifest", "m.json", "--output", "out_smf"])
            rc_ssids = run_wsl.main(["--solver", "ssids", "--manifest", "m.json", "--output", "out_ssids"])
            rc_default = run_wsl.main(["--manifest", "m.json", "--output", "out_def"])

        self.assertEqual(rc_smf, 0)
        self.assertEqual(rc_ssids, 0)
        self.assertEqual(rc_default, 0)
        self.assertEqual(len(recorded_cmds), 3)

        smf_cmd, ssids_cmd, def_cmd = recorded_cmds
        self.assertEqual(smf_cmd[smf_cmd.index("--runner") + 1], "wsl-smf")
        self.assertEqual(smf_cmd[smf_cmd.index("--solvers") + 1], "smf")
        self.assertEqual(ssids_cmd[ssids_cmd.index("--runner") + 1], "wsl")
        self.assertEqual(ssids_cmd[ssids_cmd.index("--solvers") + 1], "ssids")
        self.assertEqual(def_cmd[def_cmd.index("--runner") + 1], "wsl")
        self.assertEqual(def_cmd[def_cmd.index("--solvers") + 1], "ssids")

    def test_run_py_wsl_smf_dispatches_through_wsl_smf_command(self):
        import json
        import os
        import subprocess
        import tempfile
        from unittest import mock

        import run

        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": [{"matrix_id": "c1", "mtx_path": "x.mtx"}]}))
            dispatched = []

            def fake_run(cmd, **kw):
                dispatched.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.dict(os.environ, {"SMF_WSL_SMF_BINARY": "/home/u/bench_replay"}), \
                 mock.patch.object(run.subprocess, "run", side_effect=fake_run):
                run.main([
                    "--manifest", str(manifest),
                    "--output", str(td / "out"),
                    "--runner", "wsl-smf",
                    "--solvers", "smf",
                    "--trials", "1",
                    "--warmups", "0",
                    "--allow-short-trials",
                    "--threads", "4",
                    "--build-dir", str(td / "nobuild"),
                ])
            self.assertEqual(len(dispatched), 1)
            cmd = dispatched[0]
            self.assertIn("/home/u/bench_replay", cmd)
            self.assertIn("OMP_NUM_THREADS=4", cmd)
            self.assertNotIn("OMP_CANCELLATION=TRUE", cmd)
            self.assertNotIn("OMP_PROC_BIND=TRUE", cmd)

    def test_run_py_wsl_runner_keeps_smf_native(self):
        import json
        import os
        import subprocess
        import tempfile
        from unittest import mock

        import run

        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            manifest = td / "m.json"
            manifest.write_text(json.dumps({"cases": [{"matrix_id": "c1", "mtx_path": "x.mtx"}]}))
            native_bin = td / "bench_replay.exe"
            native_bin.write_bytes(b"")
            dispatched = []

            def fake_run(cmd, **kw):
                dispatched.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.dict(os.environ, {"SMF_WSL_SSIDS_DRIVER": "/d/ssids"}), \
                 mock.patch.object(run.subprocess, "run", side_effect=fake_run):
                run.main([
                    "--manifest", str(manifest),
                    "--output", str(td / "out"),
                    "--runner", "wsl",
                    "--solvers", "smf,ssids",
                    "--trials", "1",
                    "--warmups", "0",
                    "--allow-short-trials",
                    "--binary", str(native_bin),
                ])
            self.assertEqual(len(dispatched), 2)
            native_cmds = [c for c in dispatched if str(c[0]) == str(native_bin)]
            wsl_cmds = [c for c in dispatched if str(c[0]) == "wsl.exe"]
            self.assertEqual(len(native_cmds), 1)
            self.assertEqual(len(wsl_cmds), 1)
            self.assertIn("/d/ssids", wsl_cmds[0])


if __name__ == "__main__":
    unittest.main()
