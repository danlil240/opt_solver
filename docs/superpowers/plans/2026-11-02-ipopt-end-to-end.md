# IPOPT End-to-End Public Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a reproducible IPOPT end-to-end benchmark of smf (the `smf_ma97` plugin) against MA27, MA57, MA97 and MUMPS on three public sets (CUTEst, COPS 3.0, PGLib-OPF v23.07). The report is scored by shifted geometric mean, solved counts and iterations, and the instance list and scoring rules are frozen before the first full smf run.

**Architecture:**
- One runner (`tools/bench/e2e_run.py`) drives every set through the same IPOPT build with every linear solver. COPS and PGLib are AMPL `.nl` files run by the `ipopt` executable. CUTEst runs through a pycutest + cyipopt driver (`tools/bench/cutest_ipopt.py`) that prints the same IPOPT log.
- Records are JSONL, one per (instance, solver, threads, round). One shared parser (`tools/bench/ipopt_log.py`) reads IPOPT's wall-clock timers. A scorer (`tools/bench/e2e_score.py`) turns the records into the report tables and performance profiles.
- The board runs on Linux (WSL2 Ubuntu or a native Linux host), because CUTEst needs a Fortran toolchain, as SSIDS already does (`tools/bench/wsl_ssids.md`). One machine and one set of binaries make one report.

**Tech Stack:** Python 3.12 (conda env `smfe2e`: ipopt 3.14.20, cyipopt, pyomo, gridx-egret 0.6.2, pycutest 1.8.2), CUTEst + SIFDecode (gfortran), GCC/CMake build of smf with MKL, licensed CoinHSL 2024.5.15 `libhsl.so`, GoogleTest, unittest.

**Spec:** the recommendation at `https://claude.ai/artifact/Dvh52kmg9CwrigE8smPqUL` (plan 3), `docs/roadmap.md` Horizon 3 (a) and Horizon 1 items 1 and 4 (benchmark against MA57/MA97, keep a held-out set), and the write-ups `docs/performance/2026-10-26-ipm-mode.md` (corpus, gates, measurement notes), `docs/performance/2026-10-25-large-scale-benchmark.md` (first e2e table, MUMPS timer caveat) and `docs/performance/triage-pglib-notes.md` (PGLib e2e vs MA27/MA57).

## Background (what the spec and the code say today)

- `tools/bench/ipopt_kkt.py e2e` already runs IPOPT per linear solver on Pyomo models (`hard_models.MODELS`: 8 PGLib, 8 COPS; plus the 7 original models). Its gaps for a public board:
  - It is Windows-only: `smf_ma97.dll`, `ipopt.exe` and `COINHSL_BIN` are hard-coded.
  - `grab()` reads the **first** number of IPOPT's timing lines, which is CPU time, not wall time.
  - It has one run per call, with no rounds, no interleaving and no provenance.
- `tools/bench/ipm_gates.py` has `SUCCESS_STATUSES`, `succeeded()`, `objective_differs()` and `merge_repeats()`. `tools/bench/score.py` has `shifted_geomean()`, `performance_profile()` and `write_profile_csv()`. Reuse them; do not copy them.
- `solver/src/smf_ma97_plugin.cpp` never sets `Control::num_threads`, so smf inside IPOPT is always serial, and the plugin prints no build identity.
- The corpus is contaminated. These models were used for tuning and are marked `seen` in the report:
  - the 16 `ipopt_hard` models and the 7 `ipopt_kkt` models;
  - the CUTEst problems whose SuiteSparse KKT matrices are on the front-tier board: AUG3DCQP, BLOCKQP1, CONT-201, CONT-300, CVXQP3, NCVXQP1/3/5/7/9.
  The headline is the `unseen` split, as the roadmap's held-out rule asks.
- Known going in, so the report must not hide it: smf was slower than MA27 inside IPOPT on PGLib before PR #17/#18 (case2869 factorization 1.26 s vs 0.63 s, `triage-pglib-notes.md`). It had a robustness gap on cops_chain_800 and cops_elec_200 (`2026-10-26-ipm-mode.md`).

## Global Constraints

- **IPOPT** 3.14.20 for every solver. The same binary and the same options, except the linear-solver lines below. No per-solver tuning: each interface keeps IPOPT's defaults (`ma27_pivtol`, `ma97_*`, `mumps_*`, ...).
- **Common IPOPT options:** `max_iter 3000`, `max_wall_time 900`, `print_level 5`, `print_timing_statistics yes`. The harness kills a process at 960 s.
- **Solver ids and options:**
  - `smf` = `linear_solver ma97` + `hsllib <libsmf_ma97.so>`;
  - `ma27`, `ma57`, `ma97` = `linear_solver <id>` + `hsllib $SMF_HSL_LIBRARY` (one licensed `libhsl.so` for all three);
  - `mumps` = `linear_solver mumps` with **no** `hsllib`. This is how an open-source user runs it, and the report says so.
- **Rival set for "best rival":** `ma27, ma57, ma97, mumps`.
- **Threads:**
  - The 1-thread board is the headline. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `SMF_NUM_THREADS` are all set to T for every run.
  - The 8-thread board is secondary (Task 11). MA27 is reported at 1 thread only.
- **Environment:** the runner removes every inherited `SMF_*` variable except `SMF_NUM_THREADS` and `SMF_PLUGIN_STATS=2`, so a stray `SMF_IPM_*` cannot leak into a board.
- **Rounds:**
  - 3 rounds. The solver order rotates by one position per instance and per round, as in `tools/bench/ab_run.py`.
  - A pair that times out in round 1 is not repeated.
  - Timings are medians over rounds (`ipm_gates.merge_repeats`).
- **Metrics** (seconds are wall clock, from IPOPT's own lines):
  - `alg_s` = "Total seconds in IPOPT (w/o function evaluations)". This is the headline: it does not depend on the modelling layer (ASL vs Python).
  - `overall_s` = the `wall:` value of `OverallAlgorithm`.
  - `linsys_s` = the sum of the `wall:` values of `LinearSystemSymbolicFactorization`, `LinearSystemFactorization` and `LinearSystemBackSolve`. The sum is solver-neutral even though IPOPT does not split MUMPS's factorization out.
  - `iterations`.
- **Scoring constants:** `SHIFT_ALG_S = 1.0`, `SHIFT_LINSYS_S = 0.1`, `SHIFT_ITERS = 10`. A failure counts as `TIME_LIMIT_S = 900` in per-solver means.
  - Solved = the status starts with `ipm_gates.SUCCESS_STATUSES`.
  - Different optimum = `ipm_gates.objective_differs(obj, ref, rel=1e-6)`. It is still counted as solved, and it is listed.
- **Pre-registration:** `benchmarks/manifests/ipopt_e2e.json` and the scoring code are committed before the first full-tier smf run. They do not change after it, except to fix harness bugs, each listed in the write-up.
- **Measurement:**
  - One machine and one binary per solver per board.
  - Runs at `nice -n 10`, one at a time. First check that no `ipopt`, `python` driver or `bench_replay` is running.
  - Absolute times are never compared across boards, only ratios within one board (`2026-10-26-ipm-mode.md`, "Measurement notes").
- **Files:** fetched and decoded data go to gitignored `benchmarks/fixtures/` or `benchmarks/results/`. Committed results are added with `git add -f`, as for the existing `benchmarks/results/*.json`.
- **Tests:** Python tests run from `tools/bench` with `python -m unittest <module> -v`. C++ tests run through ctest in the build directory.

## Review Focus

1. **Timers at T > 1:** IPOPT's timing lines put CPU time first and `wall:` last. A parser that takes the first number reports the sum of thread CPU times on the 8-thread board. Expected: every time in a record is wall clock. Task 1 test: `test_parses_wall_not_cpu`.
2. **CUTEst sparsity patterns that change between calls:** pycutest returns sparse Hessians and Jacobians per point. IPOPT requires a fixed structure, and smf reuses its analysis. Expected: the pattern is fixed at the first evaluation, and any later entry outside it stops the run as a `harness` outcome. Values are never silently dropped. Task 6 test: `test_pattern_miss_raises`.
3. **IPOPT crashes, kills and "no EXIT line":** a segfault, a timeout kill or a truncated log must become a `crash` or `timeout` record that scores as a failure. It must never become a missing row, which would shrink the denominator. Task 8 tests: `test_crash_is_recorded`, `test_timeout_is_recorded`.
4. **Different local optimum:** a solver that converges to a different local minimum is "solved" by IPOPT's status, but is not comparable as a time. Expected: it counts in the solved totals and is listed with its objective gap. The reference objective is MA27's, or the median of the solved objectives when MA27 failed. Task 9 test: `test_reference_objective_fallback`.
5. **An incomplete board or an instance nobody solves:** an instance no solver solves is left out of the means but listed. A board missing an (instance, solver) pair is refused unless `--allow-incomplete` is given, and then the gap is printed. Task 9 tests: `test_unsolved_by_all_excluded_and_listed`, `test_incomplete_board_refused`.

---

### Task 1: Shared IPOPT log parser (wall-clock timers)

**Files:**
- Create: `tools/bench/ipopt_log.py`
- Create: `tools/bench/testdata/ipopt_3.14.20_case118_ma27.log`: a trimmed copy of a real 3.14.20 log from `benchmarks/results/ipopt_work/e2e/pglib_opf_case118_ieee_ma27_t1/ipopt.log`. Keep the banner, three iteration lines, the summary block and the full "Timing Statistics" block.
- Create: `tools/bench/test_ipopt_log.py`
- Modify: `tools/bench/ipopt_kkt.py` (`run_ipopt`, lines 303-345: replace `grab` with `ipopt_log.parse_log`, keeping the row keys it returns today)

**Interfaces:**
- Produces: `parse_log(text: str) -> dict` with keys `ipopt_version: str|None`, `status: str` ("no EXIT line" when absent), `iterations: int|None`, `objective: float|None`, `n_vars: int|None`, `n_eq: int|None`, `n_ineq: int|None`, `alg_s: float|None`, `func_eval_s: float|None`, `overall_s: float|None`, `linsys_symbolic_s`, `linsys_factor_s`, `linsys_backsolve_s: float|None`, `linsys_s: float|None` (the sum; None if all three are missing).
- Produces: `timer_wall(text: str, name: str) -> float|None` returns the `wall:` value of the line `<name>....: cpu (sys: s wall: w)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fixture_fields(self):
    r = ipopt_log.parse_log(FIXTURE.read_text())
    self.assertEqual(r["ipopt_version"], "3.14.20")
    self.assertTrue(r["status"].startswith("Optimal Solution Found"))
    self.assertEqual(r["iterations"], 26)
    self.assertAlmostEqual(r["linsys_s"], r["linsys_symbolic_s"] + r["linsys_factor_s"] + r["linsys_backsolve_s"])
    self.assertLessEqual(r["alg_s"], r["overall_s"] + 1e-3)

def test_parses_wall_not_cpu(self):
    line = "LinearSystemFactorization...........:      8.000 (sys:      0.100 wall:      1.250)\n"
    self.assertEqual(ipopt_log.timer_wall(line, "LinearSystemFactorization"), 1.25)

def test_missing_exit_line(self):
    self.assertEqual(ipopt_log.parse_log("This is Ipopt version 3.14.20\n")["status"], "no EXIT line")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd tools/bench && python -m unittest test_ipopt_log -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ipopt_log'`

- [ ] **Step 3: Implement `parse_log` and `timer_wall` in `tools/bench/ipopt_log.py`, then switch `ipopt_kkt.run_ipopt` to them**

`run_ipopt` keeps its output keys: `linear_system_factorization_s` comes from `linsys_factor_s`, and so on. They are now wall clock. Note that in the module docstring.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd tools/bench && python -m unittest test_ipopt_log test_ipm_gates test_plugin_trace -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/bench/ipopt_log.py tools/bench/test_ipopt_log.py tools/bench/testdata/ipopt_3.14.20_case118_ma27.log tools/bench/ipopt_kkt.py
git commit -m "bench(e2e): shared IPOPT log parser reading wall-clock timers"
```

### Task 2: Plugin thread count and build identity

**Files:**
- Modify: `solver/src/smf_ma97_plugin.cpp`:
  - `apply_ma97_control` sets `out_ctrl.num_threads`;
  - new exports after `ma97_finalise_d`;
  - the per-factorization trace (around line 651) gains `threads=%d`.
- Modify: `solver/CMakeLists.txt` (the `smf_ma97` target, about line 612): `execute_process(git rev-parse --short=12 HEAD)` into a `SMF_GIT_SHA` compile definition ("unknown" when git fails), plus `SMF_BLAS_BACKEND_NAME`.
- Modify: `solver/tests/test_ma97_plugin_abi.cpp`
- Modify: `tools/bench/plugin_trace.py`: optional `threads=` capture in `_LINE`, and the `threads` key in `parse_trace`.
- Test: `tools/bench/test_plugin_trace.py`

**Interfaces:**
- Produces (C ABI, plugin): `extern "C" int smf_ma97_num_threads(void)` returns the count `apply_ma97_control` would use now. It reads `SMF_NUM_THREADS`, then `OMP_NUM_THREADS`, else 1, clamped to [1, 256], on every call (no static caching).
- Produces: `extern "C" const char* smf_ma97_build_info(void)` returns `"smf_ma97 git=<sha> blas=<backend>"`.
- Produces: when `SMF_PLUGIN_STATS` is set, the first `ma97_analyse_d` prints one line, `[smf_ma97] build: git=<sha> blas=<backend> threads=<k>`.
- Produces (Python): `parse_trace(text)["threads"]: int|None`, the `threads=` of the last factor line, and `plugin_build(text: str) -> dict|None` in `plugin_trace.py`, giving `{"git", "blas", "threads"}` from the build line.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(Ma97PluginAbi, NumThreadsFromEnv) {
    // Plugin p loaded as in the existing fixtures; resolve "smf_ma97_num_threads".
    set_env("SMF_NUM_THREADS", "");  set_env("OMP_NUM_THREADS", "6");
    EXPECT_EQ(num_threads(), 6);
    set_env("SMF_NUM_THREADS", "3");
    EXPECT_EQ(num_threads(), 3);
    set_env("SMF_NUM_THREADS", "0");
    EXPECT_EQ(num_threads(), 1);
}
TEST(Ma97PluginAbi, BuildInfoNamesGitAndBlas) {
    std::string s = build_info();
    EXPECT_NE(s.find("git="), std::string::npos);
    EXPECT_NE(s.find("blas="), std::string::npos);
}
```

```python
def test_threads_and_build_line(self):
    text = ("[smf_ma97] build: git=6623dfb0a1b2 blas=mkl threads=8\n"
            "[smf_ma97] factor 1: corr=0 delays=0 static=0 pos=3 neg=2 zero=0 L=10 t=0.001 threads=8\n")
    self.assertEqual(plugin_trace.parse_trace(text)["threads"], 8)
    self.assertEqual(plugin_trace.plugin_build(text), {"git": "6623dfb0a1b2", "blas": "mkl", "threads": 8})
```

`set_env` is a small helper in the test file (`_putenv_s` on Windows, `setenv` elsewhere).

- [ ] **Step 2: Run them to verify they fail**

Run: `cmake --build build --target test_ma97_plugin_abi && ctest --test-dir build -R Ma97PluginAbi --output-on-failure`, then `cd tools/bench && python -m unittest test_plugin_trace -v`
Expected: FAIL. The new symbols do not resolve (null function pointer), and the `threads` key is missing.

- [ ] **Step 3: Implement the two exports, the `num_threads` assignment, the build line and the trace field; extend `_LINE` and `parse_trace`, and add `plugin_build`**

Every existing trace field keeps its position, and `threads=` goes after `t=`, so old logs still parse.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `ctest --test-dir build -R "Ma97Plugin|IpoptPluginRuntime" --output-on-failure` and `cd tools/bench && python -m unittest test_plugin_trace -v`
Expected: PASS. The existing `Ma97PluginRuntime` and `IpoptPluginRuntime` stay green.

- [ ] **Step 5: Commit**

```bash
git add solver/src/smf_ma97_plugin.cpp solver/CMakeLists.txt solver/tests/test_ma97_plugin_abi.cpp tools/bench/plugin_trace.py tools/bench/test_plugin_trace.py
git commit -m "feat(plugin): SMF_NUM_THREADS/OMP_NUM_THREADS thread count and build identity"
```

### Task 3: Linux benchmark toolchain

**Files:**
- Create: `tools/bench/e2e_environment.yml`. Conda env `smfe2e`, with exact pins: `ipopt=3.14.20`, `cyipopt`, `pyomo`, `numpy`, `scipy`, `mkl`, `mkl-devel`, `cmake`, `ninja`, plus pip entries `gridx-egret==0.6.2` and `pycutest==1.8.2`. Pin the cyipopt and pyomo versions that conda resolves on the day, written into the file.
- Create: `solver/benchmarks/wsl/build_e2e.sh`. It is idempotent and modelled on `solver/benchmarks/wsl/build.sh`, and does four things:
  1. builds SIFDecode and CUTEst at pinned release tags (`SIFDECODE_VERSION` and `CUTEST_VERSION` variables, defaults written in the script) into `$HOME/cutest-install`;
  2. clones the SIF problem collection at a pinned commit into `$HOME/sif` (`MASTSIF`);
  3. configures smf with `-DSMF_USE_MKL=ON -DSMF_MKL_THREADING=intel -DSMF_BUILD_MA97_PLUGIN=ON -DCMAKE_BUILD_TYPE=Release` into `<repo>/build-linux`;
  4. builds it.
- Create: `tools/bench/e2e_linux.md`: the one-time setup, the environment variables and the USER ACTION for CoinHSL.
- Modify: `tools/bench/ipopt_kkt.py` (lines 38-49, `BUILD_REL`/`COINHSL_BIN`/`IPOPT_EXE`, and `solver_options`). These read `SMF_E2E_IPOPT`, `SMF_E2E_PLUGIN` and `SMF_HSL_LIBRARY` when they are set. The current Windows values stay as fallbacks.

**Interfaces:**
- Produces the environment contract that Tasks 6, 8 and 10 read:
  - `SMF_E2E_IPOPT`: the `ipopt` AMPL executable;
  - `SMF_E2E_PYTHON`: the `smfe2e` python;
  - `SMF_E2E_PLUGIN`: `<repo>/build-linux/libsmf_ma97.so`;
  - `SMF_HSL_LIBRARY`: the licensed `libhsl.so`;
  - `CUTEST`, `SIFDECODE`, `MASTSIF`, `ARCHDEFS`, `MYARCH`: as pycutest 1.8.2 documents them.

- [ ] **Step 1: USER ACTION (blocking for the MA27/MA57/MA97 columns only).** The user downloads the Linux x86_64 binary of the licensed CoinHSL 2024.5.15, the same release as the Windows `libhsl.dll` in `build-deps/`, and sets `SMF_HSL_LIBRARY` to its `lib/libhsl.so`. Until then Tasks 4-9 proceed, and smoke runs use `smf` and `mumps` only.
- [ ] **Step 2: Write `e2e_environment.yml` and `build_e2e.sh`; run `bash solver/benchmarks/wsl/build_e2e.sh`.** Expected: the script exits 0, `build-linux/libsmf_ma97.so` exists, and `python -c "import pycutest, cyipopt, egret"` succeeds in `smfe2e`.
- [ ] **Step 3: Run the Linux test suite.** Run: `ctest --test-dir build-linux --output-on-failure -j4`. Expected: PASS, including `Ma97PluginAbi.NumThreadsFromEnv`. Record any test that is Windows-only and skipped in `e2e_linux.md`.
- [ ] **Step 4: Smoke every solver on one model.** Run `python tools/bench/ipopt_kkt.py e2e --models pglib_opf_case118_ieee --solvers smf,ma27,ma57,ma97,mumps` with the Linux paths (Task 8 replaces this call; here `ipopt_kkt`'s paths are overridden by the `SMF_E2E_*` variables, read at the top of `ipopt_kkt.py` with the Windows values as fallbacks). Expected: all five runs end "Optimal Solution Found." in 26 iterations (the iteration count on Windows, `ipopt_hard_baseline_t1.json`), and `linsys_s > 0` for mumps.
- [ ] **Step 5: Commit**

```bash
git add tools/bench/e2e_environment.yml solver/benchmarks/wsl/build_e2e.sh tools/bench/e2e_linux.md tools/bench/ipopt_kkt.py
git commit -m "bench(e2e): Linux toolchain for the public IPOPT benchmark (CUTEst, conda env, smf plugin build)"
```

### Task 4: PGLib-OPF v23.07 public set

**Files:**
- Modify: `tools/bench/fetch_pglib.py`. Add the set `E2E_CASES`: the full v23.07 list of typical cases plus their `__api` and `__sad` variants with ≤ 10,000 buses, and the typical case alone above 10,000 buses. Leave out `case78484_epigrids*`, which exceeds the 900 s limit for every solver. Add `--set {hard,e2e}` (default `hard` = today's `CASES`, unchanged).
- Modify: `tools/bench/hard_models.py` (`acopf_pglib`: case names resolve for any `__api`/`__sad` case; today it only knows the 8)
- Create: `tools/bench/test_fetch_pglib.py`

**Interfaces:**
- Produces: `fetch_pglib.E2E_CASES: dict[str, tuple[str, str]]`, the same shape as `CASES` (name → (subdir, sha256)). The subdir is `""`, `"api"` or `"sad"` from the name suffix. Hashes are frozen with `--print-hashes --set e2e`.
- Produces: `fetch_pglib.bus_count(name: str) -> int` (from `case<digits>`).

- [ ] **Step 1: Write the failing tests**

```python
def test_e2e_superset_of_hard(self):
    for k, v in fetch_pglib.CASES.items():
        self.assertEqual(fetch_pglib.E2E_CASES[k], v)

def test_variant_rule(self):
    for name, (sub, sha) in fetch_pglib.E2E_CASES.items():
        self.assertEqual(sub, "api" if name.endswith("__api") else "sad" if name.endswith("__sad") else "")
        self.assertRegex(sha, r"^[0-9a-f]{64}$")
        if sub:
            self.assertLessEqual(fetch_pglib.bus_count(name), 10000)
    self.assertNotIn("pglib_opf_case78484_epigrids", fetch_pglib.E2E_CASES)
    self.assertIn("pglib_opf_case30000_goc", fetch_pglib.E2E_CASES)
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd tools/bench && python -m unittest test_fetch_pglib -v`. Expected: FAIL with `AttributeError: ... 'E2E_CASES'`.
- [ ] **Step 3: Implement `E2E_CASES`, `bus_count` and `--set`.** Take the case list from the tag's `BASELINE.md`. Freeze the hashes: `python tools/bench/fetch_pglib.py --set e2e --print-hashes`, then paste the output.
- [ ] **Step 4: Verify.** Run `python tools/bench/fetch_pglib.py --set e2e` (expected: every line `ok` or `fetched`, exit 0), then `python -m unittest test_fetch_pglib test_hard_models -v` (expected: PASS).
- [ ] **Step 5: Commit**

```bash
git add tools/bench/fetch_pglib.py tools/bench/hard_models.py tools/bench/test_fetch_pglib.py
git commit -m "bench(e2e): PGLib-OPF v23.07 public set (typical, api, sad) with frozen hashes"
```

### Task 5: COPS 3.0 public set (the remaining problems without external data)

**Files:**
- Create: `tools/bench/cops_models.py`. Pyomo ports of `bearing`, `camshape`, `channel`, `gasoil`, `marine`, `methanol`, `minsurf`, `robot` and `torsion`. Follow the formulations and starting points of COPS 3.0 (ANL/MCS-TM-273), cross-checked against COPSBenchmark.jl, in the style of the `hard_models.py` COPS section. `robot` reuses `ipopt_kkt.robot_arm`, not a second port.
- Create: `tools/bench/test_cops_models.py`

**Interfaces:**
- Produces: `cops_models.COPS_SIZES: dict[str, tuple[int, int, int]]`, the three discretization levels per problem from the COPS 3.0 report's tables. It covers the 8 problems already in `hard_models.COPS` and the 9 new ones.
- Produces: `cops_models.COPS_REFERENCE: dict[tuple[str, int], float]`, the report's optimal objective for the smallest level of each problem.
- Produces: `cops_models.MODELS: dict[str, tuple[builder, int]]`, with names `cops_<problem>_<size>` and the same shape as `hard_models.MODELS`.
- Consumes: `hard_models.COPS` builders (unchanged).

- [ ] **Step 1: Write the failing tests**

```python
def test_three_levels_per_problem(self):
    self.assertEqual(len(cops_models.COPS_SIZES), 17)
    for p, levels in cops_models.COPS_SIZES.items():
        self.assertEqual(list(levels), sorted(levels))
        for s in levels:
            self.assertIn(f"cops_{p}_{s}", cops_models.MODELS)

@unittest.skipUnless(shutil.which(os.environ.get("SMF_E2E_IPOPT", "ipopt")), "ipopt not found")
def test_smallest_level_matches_report(self):
    for (p, s), ref in cops_models.COPS_REFERENCE.items():
        obj = solve_with_mumps(cops_models.MODELS[f"cops_{p}_{s}"])  # helper: write .nl, run ipopt, ipopt_log.parse_log
        self.assertLessEqual(abs(obj - ref), 1e-6 * max(1.0, abs(ref)), f"{p}_{s}")
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd tools/bench && python -m unittest test_cops_models -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'cops_models'`.
- [ ] **Step 3: Port the 9 problems and fill `COPS_SIZES`, `COPS_REFERENCE` and `MODELS`.** A problem whose smallest level does not reach its reference value within 1e-6 is a porting bug: fix the port, never the reference.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `cd tools/bench && python -m unittest test_cops_models test_hard_models -v`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add tools/bench/cops_models.py tools/bench/test_cops_models.py
git commit -m "bench(e2e): COPS 3.0 ports (bearing..torsion) with three levels and report objectives"
```

### Task 6: CUTEst driver and frozen selection

**Files:**
- Create: `tools/bench/cutest_ipopt.py`, a one-problem-per-process driver: `python -I cutest_ipopt.py <PROBLEM> --opt-file ipopt.opt`. IPOPT's log goes to stdout, so `ipopt_log.parse_log` reads it unchanged.
- Create: `tools/bench/cutest_select.py`, a one-off selection that writes `benchmarks/manifests/cutest_selection.json`.
- Create: `tools/bench/test_cutest_ipopt.py`

**Interfaces:**
- Produces: `class FixedPattern(rows: np.ndarray, cols: np.ndarray)` with `values(sparse) -> np.ndarray`. It maps a scipy COO/CSR matrix's entries into the fixed pattern order, summing duplicates. It raises `PatternMiss(row, col)` for any entry outside the pattern (Review Focus 2). The Hessian pattern is the lower triangle of the first `sphess` call at `x0` with multipliers 1; the Jacobian pattern comes from the first `scons(x0, gradient=True)`.
- Produces: `class CutestNLP` with cyipopt's callbacks (`objective`, `gradient`, `constraints`, `jacobian`, `jacobianstructure`, `hessian`, `hessianstructure`), built from a `pycutest` problem. The driver exits with code 3 and prints `HARNESS: pattern miss <row> <col>` on `PatternMiss`.
- Produces: `select(props: list[dict]) -> list[str]`, the frozen rule. It keeps a problem when its default-SIF-parameter `n + m ≥ 1000` and `n ≤ 1_000_000`. `props` items are `{"name", "n", "m"}` read from decoded problems. Only problems with a variable size (`'V'` in the classification) or a fixed `n + m ≥ 1000` are decoded.
- Selection file schema: `{"cutest_version", "sifdecode_version", "mastsif_commit", "problems": [{"name", "n", "m", "sif_sha256"}]}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pattern_maps_and_sums_duplicates(self):
    pat = cutest_ipopt.FixedPattern(np.array([0, 1, 1]), np.array([0, 0, 1]))
    m = scipy.sparse.coo_matrix(([1.0, 2.0, 0.5, 3.0], ([0, 1, 1, 1], [0, 0, 0, 1])), shape=(2, 2))
    np.testing.assert_allclose(pat.values(m), [1.0, 2.5, 3.0])

def test_pattern_miss_raises(self):
    pat = cutest_ipopt.FixedPattern(np.array([0]), np.array([0]))
    with self.assertRaises(cutest_ipopt.PatternMiss):
        pat.values(scipy.sparse.coo_matrix(([1.0], ([1], [0])), shape=(2, 2)))

def test_select_rule(self):
    props = [{"name": "A", "n": 999, "m": 0}, {"name": "B", "n": 500, "m": 500}, {"name": "C", "n": 2_000_000, "m": 0}]
    self.assertEqual(cutest_select.select(props), ["B"])

@unittest.skipUnless(os.environ.get("CUTEST"), "CUTEst not installed")
def test_hs71_solves(self):
    r = run_driver("HS71", solver="mumps")  # helper: subprocess + ipopt_log.parse_log
    self.assertTrue(r["status"].startswith("Optimal Solution Found"))
    self.assertAlmostEqual(r["objective"], 17.0140173, places=6)
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd tools/bench && python -m unittest test_cutest_ipopt -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'cutest_ipopt'`.
- [ ] **Step 3: Implement `FixedPattern`, `PatternMiss`, `CutestNLP`, the driver `main`, and `cutest_select.select` plus its CLI.** The driver reads the same option file the runner writes (Task 8), so options are identical across sets. Bounds of ±1e20 map to IPOPT's infinity.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `cd tools/bench && python -m unittest test_cutest_ipopt -v`. Expected: PASS, with `test_hs71_solves` running in `smfe2e`.
- [ ] **Step 5: Run the selection once and freeze it.** Run: `nice -n 10 python -I tools/bench/cutest_select.py --out benchmarks/manifests/cutest_selection.json`. Expected: it exits 0 and prints the kept count, and every kept problem has a `sif_sha256`.
- [ ] **Step 6: Commit**

```bash
git add tools/bench/cutest_ipopt.py tools/bench/cutest_select.py tools/bench/test_cutest_ipopt.py
git add -f benchmarks/manifests/cutest_selection.json
git commit -m "bench(e2e): CUTEst driver (pycutest + cyipopt, fixed patterns) and frozen selection"
```

### Task 7: Public-set manifest (pre-registration)

**Files:**
- Create: `tools/bench/e2e_sets.py`
- Create: `benchmarks/manifests/ipopt_e2e.json` (generated, committed)
- Create: `tools/bench/test_e2e_sets.py`

**Interfaces:**
- Consumes: `fetch_pglib.E2E_CASES`, `cops_models.MODELS`, `cops_models.COPS_SIZES`, `benchmarks/manifests/cutest_selection.json`, `hard_models.MODELS`, `ipopt_kkt.MODELS`, and the `matrix_id`s of `benchmarks/manifests/fronttier.json`.
- Produces: `build_manifest() -> dict` and the file `{"schema": "ipopt_e2e/1", "constants": {...the Global Constraints' scoring constants and IPOPT options...}, "instances": [...]}`. Each instance is `{"id", "set": "cutest"|"cops"|"pglib", "split": "seen"|"unseen", "tier": "smoke"|"full", "source": {...}, "source_sha256"}`.
- `source` takes one of two forms:
  - for `.nl` sets: `{"builder": "<module>.<name>", "arg"}`;
  - for CUTEst: `{"problem"}`.
- Produces: `is_seen(instance_id: str, set_name: str, fronttier_ids: set[str]) -> bool`. The rule:
  - an id in `hard_models.MODELS` or `ipopt_kkt.MODELS` is seen;
  - a COPS problem with the same problem name and size as a `hard_models.COPS` entry is seen;
  - a CUTEst name is seen when its lowercase form is in `fronttier_ids`.
- The smoke tier is the first 4 ids by sorted id in each (set, split) pair: at most 24 instances, deterministic.

- [ ] **Step 1: Write the failing tests**

```python
def test_seen_rule(self):
    ft = {"cvxqp3", "ncvxqp9", "cont-201"}
    self.assertTrue(e2e_sets.is_seen("CVXQP3", "cutest", ft))
    self.assertTrue(e2e_sets.is_seen("CONT-201", "cutest", ft))
    self.assertFalse(e2e_sets.is_seen("HS71", "cutest", ft))
    self.assertTrue(e2e_sets.is_seen("pglib_opf_case9241_pegase", "pglib", ft))
    self.assertFalse(e2e_sets.is_seen("pglib_opf_case9241_pegase__api", "pglib", ft))
    self.assertTrue(e2e_sets.is_seen("cops_chain_800", "cops", ft))

def test_manifest_deterministic_and_unique(self):
    a, b = e2e_sets.build_manifest(), e2e_sets.build_manifest()
    self.assertEqual(a, b)
    ids = [i["id"] for i in a["instances"]]
    self.assertEqual(len(ids), len(set(ids)))
    self.assertEqual(a["constants"]["SHIFT_ALG_S"], 1.0)
    self.assertLessEqual(sum(i["tier"] == "smoke" for i in a["instances"]), 24)
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd tools/bench && python -m unittest test_e2e_sets -v`. Expected: FAIL with `ModuleNotFoundError`.
- [ ] **Step 3: Implement `is_seen`, `build_manifest` and the CLI.** The CLI is `python tools/bench/e2e_sets.py --out benchmarks/manifests/ipopt_e2e.json`. `source_sha256` is the `.m` hash for PGLib, the SIF hash for CUTEst, and the sha256 of the written `.nl` for COPS (the builders are deterministic).
- [ ] **Step 4: Run the tests and generate the manifest.** Run: `python -m unittest test_e2e_sets -v`, then the CLI. Expected: PASS, and the CLI prints the counts per (set, split).
- [ ] **Step 5: Commit (this commit is the pre-registration point; record its sha in the write-up)**

```bash
git add tools/bench/e2e_sets.py tools/bench/test_e2e_sets.py
git add -f benchmarks/manifests/ipopt_e2e.json
git commit -m "bench(e2e): pre-registered public-set manifest (CUTEst, COPS, PGLib; seen/unseen)"
```

### Task 8: Reproducible runner

**Files:**
- Create: `tools/bench/e2e_run.py`
- Create: `tools/bench/testdata/fake_ipopt.py`. A stand-in executable that replays a canned log from `FAKE_IPOPT_LOG`. With `FAKE_IPOPT_MODE=crash` it exits -11 after printing half the log; with `sleep` it sleeps past the timeout. It also writes its received environment to `FAKE_IPOPT_ENV_OUT`.
- Create: `tools/bench/test_e2e_run.py`

**Interfaces:**
- Consumes: `ipopt_log.parse_log`, `plugin_trace.parse_trace`, `plugin_trace.plugin_build`, the manifest from Task 7, and the Task 3 environment contract.
- Produces: the CLI `python tools/bench/e2e_run.py --manifest benchmarks/manifests/ipopt_e2e.json --tier smoke|full --sets cutest,cops,pglib --solvers smf,ma27,ma57,ma97,mumps --threads 1 --rounds 3 --output <dir>`.
  - It writes `<dir>/records.jsonl`, appending one line per run. Pairs already present are skipped (resume).
  - It writes `<dir>/provenance.json` once.
- Produces functions (each one testable):
  - `solver_order(solvers: list[str], instance_index: int, round_: int) -> list[str]` rotates the list by `instance_index + round_`.
  - `run_env(base: dict, solver: str, threads: int) -> dict` scrubs `SMF_*` and sets the four thread variables and `SMF_PLUGIN_STATS=2`.
  - `ipopt_options(solver: str, constants: dict) -> list[str]` follows the Global Constraints.
  - `classify(parsed: dict, returncode: int|None, timed_out: bool) -> str` returns one of `"solved"`, `"failed"`, `"timeout"`, `"crash"` or `"harness"`. `harness` is a driver exit code of 3 or a missing `.nl`/SIF source.
  - `run_one(instance: dict, solver: str, threads: int, round_: int, workdir: Path) -> dict` returns the record.
- Record keys: `schema`, `instance_id`, `set`, `split`, `solver`, `threads`, `round`, `order_pos`, `outcome`, `status`, `iterations`, `objective`, `alg_s`, `overall_s`, `linsys_s`, `func_eval_s`, `process_wall_s`, `factorizations`, `inertia_corrections`, `log`.
- Provenance keys: `git_sha` (repo HEAD), `plugin_build` (from the first smf log), `plugin_sha256`, `hsl_sha256`, `ipopt_version`, `conda_explicit_sha256` (of `conda list --explicit`), `cpu` (`lscpu` "Model name"), `kernel` (`uname -r`), `started`.

- [ ] **Step 1: Write the failing tests**

```python
def test_rotation(self):
    s = ["smf", "ma27", "ma57", "ma97", "mumps"]
    self.assertEqual(e2e_run.solver_order(s, 0, 0), s)
    self.assertEqual(e2e_run.solver_order(s, 1, 1)[0], "ma57")

def test_env_scrub(self):
    env = e2e_run.run_env({"SMF_IPM_STATIC": "regularized", "PATH": "/bin"}, "smf", 8)
    self.assertNotIn("SMF_IPM_STATIC", env)
    self.assertEqual((env["SMF_NUM_THREADS"], env["OMP_NUM_THREADS"], env["SMF_PLUGIN_STATS"]), ("8", "8", "2"))

def test_mumps_has_no_hsllib(self):
    self.assertFalse(any(o.startswith("hsllib") for o in e2e_run.ipopt_options("mumps", CONST)))
    self.assertIn("max_wall_time 900", e2e_run.ipopt_options("smf", CONST))

def test_crash_is_recorded(self):    # fake_ipopt, FAKE_IPOPT_MODE=crash
    r = self.run_fake("crash")
    self.assertEqual(r["outcome"], "crash")

def test_timeout_is_recorded(self):  # fake_ipopt, FAKE_IPOPT_MODE=sleep, kill timeout 1 s
    r = self.run_fake("sleep", kill_s=1)
    self.assertEqual(r["outcome"], "timeout")

def test_resume_skips_done_pairs(self):
    # two invocations over the same output dir: the second appends nothing
    ...
    self.assertEqual(len(lines_after_second), len(lines_after_first))
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd tools/bench && python -m unittest test_e2e_run -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'e2e_run'`.
- [ ] **Step 3: Implement `e2e_run.py`.**
  - The `.nl` files are built once into `benchmarks/results/ipopt_work/nl/` through `ipopt_kkt.build_nl` (extended to resolve `cops_models.MODELS`), and their sha256 is checked against the manifest's `source_sha256`. A mismatch is a `harness` record.
  - The process runs under `nice -n 10`, with the 960 s kill.
  - Each run's log goes to `<dir>/logs/<instance>_<solver>_t<T>_r<round>.log`.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `cd tools/bench && python -m unittest test_e2e_run -v`. Expected: PASS.
- [ ] **Step 5: Real smoke tier, one round.** Run: `python tools/bench/e2e_run.py --manifest benchmarks/manifests/ipopt_e2e.json --tier smoke --rounds 1 --threads 1 --output benchmarks/results/ipopt_e2e_smoke`. Expected: every (instance, solver) pair has a record and no record is `harness`. `provenance.json` names the plugin git sha equal to `git rev-parse --short=12 HEAD`.
- [ ] **Step 6: Commit**

```bash
git add tools/bench/e2e_run.py tools/bench/test_e2e_run.py tools/bench/testdata/fake_ipopt.py tools/bench/ipopt_kkt.py
git commit -m "bench(e2e): interleaved, resumable IPOPT end-to-end runner with provenance"
```

### Task 9: Scorer and report renderer

**Files:**
- Create: `tools/bench/e2e_score.py`
- Create: `tools/bench/test_e2e_score.py`

**Interfaces:**
- Consumes: `records.jsonl` (Task 8), the manifest (Task 7), `ipm_gates.merge_repeats`, `ipm_gates.succeeded`, `ipm_gates.objective_differs`, `score.shifted_geomean`, `score.performance_profile` and `score.write_profile_csv`.
- Produces: `score_board(records: list[dict], manifest: dict, threads: int, allow_incomplete: bool = False) -> dict`. It raises `IncompleteBoard(missing: list[tuple[str, str]])` when a manifest instance of the selected tier and sets lacks a solver, unless `allow_incomplete`. For each `(set|"all", split|"all")` group it returns:
  - `solved[solver]`, `failed[solver]` (by outcome), `unsolved_by_all: list[str]`, `different_optimum: list[{instance, solver, objective, reference}]`;
  - `sgm_alg[solver]`: the shifted geometric mean of `alg_s` with shift `SHIFT_ALG_S`, over instances solved by at least one solver, failures at `TIME_LIMIT_S`, divided by the minimum over solvers (1.0 = best);
  - `smf_vs[rival]`: `score.shifted_geomean` of smf/rival `alg_s` pairs over instances both solved (shift 1.0);
  - `smf_vs_best_rival`: the same against the per-instance fastest solved rival;
  - the same two ratios for `linsys_s` (shift 0.1) and `iterations` (shift 10);
  - `smf_vs_best_rival_ge1s`: the ratio restricted to instances whose best-rival `alg_s ≥ 1.0`.
- Produces: `reference_objective(rows: dict[str, dict]) -> float|None`. It is MA27's objective if MA27 solved the instance, else the median objective of the solvers that solved it.
- Produces: `render(result: dict) -> str`, the Markdown tables:
  1. the headline: unseen, 1T, `smf_vs_best_rival` and solved counts per set;
  2. per-solver `sgm_alg` and solved/failed per set and split;
  3. pairwise smf vs each rival (alg, linsys, iterations);
  4. lists: unsolved by all, different optima, smf-only failures and smf-only solves.
- Produces: the CLI `python tools/bench/e2e_score.py --records <dir> --manifest benchmarks/manifests/ipopt_e2e.json --threads 1 [--tier full] [--allow-incomplete] --output <dir>/report.md --profile-csv <dir>`. It writes `profile_<set>_<split>.csv` for the `alg_s` performance profiles (`taus` = 1, 1.25, 1.5, 2, 3, 5, 10, 20, 50, 100).

- [ ] **Step 1: Write the failing tests** (synthetic records built by a `rec(instance, solver, alg_s, outcome="solved", objective=1.0, round_=0, ...)` helper)

```python
def test_shifted_means_and_failures_at_limit(self):
    recs = [rec("p1", "smf", 2.0), rec("p1", "ma27", 1.0), rec("p2", "smf", 0.0, outcome="failed"), rec("p2", "ma27", 3.0)]
    r = e2e_score.score_board(recs, MAN, threads=1)["groups"][("all", "all")]
    self.assertEqual(r["solved"], {"smf": 1, "ma27": 1})
    self.assertAlmostEqual(r["smf_vs"]["ma27"], 3.0 / 2.0)              # only p1 is solved by both
    self.assertGreater(r["sgm_alg"]["smf"], r["sgm_alg"]["ma27"])         # p2 counted at 900 s for smf

def test_reference_objective_fallback(self):
    self.assertEqual(e2e_score.reference_objective({"ma27": {"outcome": "solved", "objective": 5.0},
                                                    "smf": {"outcome": "solved", "objective": 7.0}}), 5.0)
    self.assertEqual(e2e_score.reference_objective({"ma27": {"outcome": "failed", "objective": None},
                                                    "ma57": {"outcome": "solved", "objective": 1.0},
                                                    "ma97": {"outcome": "solved", "objective": 2.0},
                                                    "smf": {"outcome": "solved", "objective": 9.0}}), 2.0)

def test_unsolved_by_all_excluded_and_listed(self):
    recs = [rec("p1", s, 1.0) for s in SOLVERS] + [rec("p9", s, 0.0, outcome="timeout") for s in SOLVERS]
    g = e2e_score.score_board(recs, MAN, threads=1)["groups"][("all", "all")]
    self.assertEqual(g["unsolved_by_all"], ["p9"])
    self.assertEqual(g["sgm_alg"]["smf"], 1.0)

def test_incomplete_board_refused(self):
    with self.assertRaises(e2e_score.IncompleteBoard):
        e2e_score.score_board([rec("p1", "smf", 1.0)], MAN, threads=1)

def test_repeats_use_median_and_flag_inconsistency(self):
    recs = [rec("p1", "smf", t, round_=k, iterations=10 + (k == 2)) for k, t in enumerate((1.0, 3.0, 2.0))] + ...
    g = e2e_score.score_board(recs, MAN, threads=1)
    self.assertIn(("p1", "smf"), g["inconsistent_repeats"])
```

- [ ] **Step 2: Run them to verify they fail.** Run: `cd tools/bench && python -m unittest test_e2e_score -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'e2e_score'`.
- [ ] **Step 3: Implement `score_board`, `reference_objective`, `render`, `IncompleteBoard` and the CLI.** Rounds are merged through `ipm_gates.merge_repeats`, after mapping `instance_id`/`solver` to the `model`/`solver` keys it groups on and adding `alg_s`, `linsys_s` and `overall_s` to the timings it takes medians of. Pass those as a parameter: add `timings: tuple[str, ...] = REPEAT_TIMINGS` to `merge_repeats`, which keeps its current default for its current callers. Pairs that are not `repeats_consistent` go in `inconsistent_repeats`.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `cd tools/bench && python -m unittest test_e2e_score test_ipm_gates test_score -v`. Expected: PASS.
- [ ] **Step 5: Score the Task 8 smoke board.** Run: `python tools/bench/e2e_score.py --records benchmarks/results/ipopt_e2e_smoke --manifest benchmarks/manifests/ipopt_e2e.json --threads 1 --tier smoke --output benchmarks/results/ipopt_e2e_smoke/report.md --profile-csv benchmarks/results/ipopt_e2e_smoke`. Expected: exit 0, and the report has the four sections.
- [ ] **Step 6: Commit**

```bash
git add tools/bench/e2e_score.py tools/bench/test_e2e_score.py tools/bench/ipm_gates.py
git commit -m "bench(e2e): scorer (shifted geomeans, solved counts, iterations, profiles) and report"
```

### Task 10: 1-thread public board, write-up and publication

**Files:**
- Create: `benchmarks/results/ipopt_e2e_public_t1/`: `records.jsonl`, `provenance.json`, `report.md`, `profile_*.csv` (committed with `git add -f`; logs stay uncommitted)
- Create: `docs/performance/<date>-ipopt-end-to-end.md`, dated the day the board finishes
- Modify: `docs/roadmap.md`: "Where we stand" and Horizon 3 (a) point to the write-up; strike "IPOPT end to end has not been re-run with the new plugin".

**Interfaces:**
- Consumes: `e2e_run.py` and `e2e_score.py` CLIs, the frozen manifest (Task 7 commit sha).

**The gate (all of it must hold before the write-up claims anything):**
1. `git log` shows the Task 7 commit before the first full-tier record's `provenance.started`.
2. There are zero `harness` outcomes on the full tier. Every `inconsistent_repeats` pair is listed in the write-up with its cause.
3. Reproducibility: re-running the smoke tier (3 rounds) from the same binaries on another day gives exactly the same solved set per solver. Each solver's `sgm_alg` stays within ±5% of the full board's smoke-tier subset.
4. No harness-only change after the pre-registration commit goes unlisted.

- [ ] **Step 1: Check the machine is idle, then run the full board.** Run: `pgrep -a "ipopt|bench_replay|cutest_ipopt" || nice -n 10 python tools/bench/e2e_run.py --manifest benchmarks/manifests/ipopt_e2e.json --tier full --threads 1 --rounds 3 --output benchmarks/results/ipopt_e2e_public_t1`. The run resumes if interrupted. Expected: it exits 0, and every (instance, solver) pair has at least one record.
- [ ] **Step 2: Score it.** Run: `python tools/bench/e2e_score.py --records benchmarks/results/ipopt_e2e_public_t1 --manifest benchmarks/manifests/ipopt_e2e.json --threads 1 --tier full --output benchmarks/results/ipopt_e2e_public_t1/report.md --profile-csv benchmarks/results/ipopt_e2e_public_t1`. Expected: exit 0 with no `IncompleteBoard`.
- [ ] **Step 3: Reproducibility re-run (gate item 3).** Run the smoke tier into `benchmarks/results/ipopt_e2e_repro_t1` and score it. Expected: identical solved sets, and `sgm_alg` within ±5% per solver of the smoke-tier rows of step 2's board. If not, find the noise source (pinning, thermal, a background process) before writing anything.
- [ ] **Step 4: Write `docs/performance/<date>-ipopt-end-to-end.md`.** Follow the existing write-ups' structure: Result, Corpus, Solvers and options, Method, Results per set, What smf loses and why, Measuring.
  - **Result:** the headline table (unseen, 1T).
  - **Corpus:** counts per set and split, and the pre-registration sha.
  - **Solvers and options:** versions from `provenance.json`, and the MUMPS-without-HSL rule.
  - **Method:** rounds, rotation, metrics and shifts.
  - **Results per set:** the seen and unseen tables, the profile plots' CSV paths, the different-optimum and failure lists.
  - **What smf loses and why:** each smf loss over 1.1× and each smf-only failure, with its plugin trace (`factorizations`, `inertia_corrections`, delays) and a pointer to the open item it maps to (`2026-10-26-ipm-mode.md`'s robustness gap, the brief's matching-phase losses).
  - **Measuring.**
  - It ends with the roadmap exit criterion stated as met or not met on the unseen split: "smf matches or beats MA57 and MA97 on IPOPT end to end".
- [ ] **Step 5: Publish.** Publish the write-up as an Artifact page (HTML built from the Markdown, with the `alg_s` performance profiles drawn from the CSVs). Link the page from the write-up's header.
- [ ] **Step 6: Commit**

```bash
git add -f benchmarks/results/ipopt_e2e_public_t1/records.jsonl benchmarks/results/ipopt_e2e_public_t1/provenance.json benchmarks/results/ipopt_e2e_public_t1/report.md benchmarks/results/ipopt_e2e_public_t1/profile_*.csv
git add docs/performance/*-ipopt-end-to-end.md docs/roadmap.md
git commit -m "docs(performance): IPOPT end-to-end public benchmark, 1 thread (CUTEst, COPS, PGLib)"
```

### Task 11: 8-thread board (secondary)

**Files:**
- Create: `benchmarks/results/ipopt_e2e_public_t8/` (same contents as Task 10's directory)
- Modify: `docs/performance/<date>-ipopt-end-to-end.md` (adds an "8 threads" section)

**Interfaces:**
- Consumes: Task 2's `SMF_NUM_THREADS` and `plugin_build` (the provenance must show `threads=8`), and the Task 8/9 CLIs with `--threads 8 --solvers smf,ma57,ma97,mumps`.

- [ ] **Step 1: Verify that threading reaches smf.** Run one smoke instance at `--threads 8` and check its log. Expected: `[smf_ma97] build: ... threads=8`, and the factor lines show `threads=8`. Then check `parse_log(...)["alg_s"] <= process_wall_s` (Review Focus 1).
- [ ] **Step 2: Run the full tier at 8 threads (3 rounds, nice, machine idle) and score it with `--threads 8 --allow-incomplete`.** The flag is needed because MA27 is absent by design. The report states that MA27 is absent. Expected: exit 0, and zero `harness` outcomes.
- [ ] **Step 3: Add the 8-thread section to the write-up.** It has the headline and the pairwise tables, plus run-to-run differences in iterations: multi-threaded runs are not bitwise deterministic (`2026-10-26-ipm-mode.md`, open items), so the section reports the iteration spread across rounds instead of flagging it. Cross-reference plan 4 (intra-front parallelism) for the losses it would address.
- [ ] **Step 4: Commit**

```bash
git add -f benchmarks/results/ipopt_e2e_public_t8/records.jsonl benchmarks/results/ipopt_e2e_public_t8/provenance.json benchmarks/results/ipopt_e2e_public_t8/report.md benchmarks/results/ipopt_e2e_public_t8/profile_*.csv
git add docs/performance/*-ipopt-end-to-end.md
git commit -m "docs(performance): IPOPT end-to-end public benchmark, 8 threads"
```

---

## Out of scope

- Packaging smf for conda-forge and pip, and offering it to CasADi and Clarabel users (Horizon 3 (a), later bullets). This plan produces the evidence those efforts cite.
- MA86, SPRAL SSIDS through IPOPT's `spral` interface, and cuDSS. Add them later as solver ids in `e2e_run.ipopt_options` without touching the scorer.
- Any smf change made to improve the board. A loss found here becomes its own plan, and the manifest stays frozen.
