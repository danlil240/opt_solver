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
