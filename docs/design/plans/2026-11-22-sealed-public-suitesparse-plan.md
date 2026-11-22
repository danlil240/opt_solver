# Sealed Public SuiteSparse Benchmark & Full-Rival Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a sealed, 100% public, strictly unseen SuiteSparse benchmark (60 matrices / 95 case-tracks) racing `smf` head-to-head against MA57, MA97, MKL PARDISO, Panua PARDISO, MUMPS, CHOLMOD, WSL2 SSIDS, and NVIDIA cuDSS on one host (Ryzen 9 9950X3D + RTX 5080).

**Architecture:** Implement a dynamically loaded `PanuaPardisoSolverAdapter` (`pardisoinit_d` / `pardiso_d`) alongside MKL PARDISO without symbol collisions; extend `score.py` to score `panua_pardiso`, exempt GPU solvers from the CPU GFLOP/s plausibility gate, and emit dual CPU-only and CPU+GPU headline summaries; build a deterministic SuiteSparse catalog fetcher over `ssstats.csv` that excludes `fronttier.json` series siblings and flags algebraically singular systems pre-seal; extend `run_wsl.py` / `run.py` for WSL2 OpenBLAS `smf` OS-parity and Panua/GPU provenance; commit the sealed manifest before running `smf` on any new matrix, then run the full blind board, post-seal `inertia_ref.py`, and publish the unfiltered report.

**Tech Stack:** C++20 (MSVC / CMake / GoogleTest), Python 3.12 (`unittest`, `numpy`, `scipy`), Intel MKL, CoinHSL 2024.5.15, Panua PARDISO 2024-06-30 (`libpardiso.dll`), MUMPS, CHOLMOD, NVIDIA cuDSS 0.8.0.10 (CUDA 13, RTX 5080), WSL2 Ubuntu (SPRAL SSIDS + OpenBLAS Linux `smf`).

**Spec:** [2026-11-22-sealed-public-suitesparse-design.md](file:///c:/Users/daniel/projects/opt_solver/docs/design/specs/2026-11-22-sealed-public-suitesparse-design.md)

## Global Constraints

- **Zero overlap with `fronttier.json`:** Exclude both exact names and same-generator series siblings (`TSOPF_FS_*`, `af_shell*`, `thermal*`, `ncvxqp*`, `cvxqp*`, `cont-*`, `boyd*`, `aug3d*`, `nd*k`, `cfd*`, `Dubcova*`, `apache*`, `stokes*`, `mario*`, `darcy*`, `bratu*`, `helm*`, `copter*`, `dawson*`, `blockqp*`) from the 45 SuiteSparse matrices in `benchmarks/manifests/fronttier.json`. Keep `Schenk_IBMNA`'s `c-*` series except `c-big`.
- **SuiteSparse index & size bounds:** Fetch from `https://www.cise.ufl.edu/research/sparse/matrices/ssstats.csv`; filter `isReal == 1`, `isBinary == 0`, `nrows == ncols`, `numerical_symmetry == 1.0`, excluding graph/combinatorial/bipartite/duplicate kinds; `20_000 <= n <= 600_000`, full symmetric `nnz <= 30_000_000`, total archive budget `<= 2500 MB`.
- **Sealing discipline:** Never run `smf` on any matrix from `public_suitesparse.json` before the pre-registered manifest commit. Never drop/edit matrices after seeing results. Do not edit `benchmarks/manifests/fronttier.json`.
- **Licensed binaries & keys:** Never commit `panua.lic`, `libpardiso.dll`, `libpardiso.so`, or `libcoinhsl.dll` to git. Extract Panua only under gitignored `build-deps/panua-pardiso/`.
- **Fair thread reporting:** Run `T = 1` and `T = 8`; `ma57` runs at `T = 1` only; `cholmod` and `cudss` run at both `T = 1` and `T = 8`. If Panua's license caps cores at `K < 8`, run Panua + `smf` at `T = 1` and `T = K` and omit Panua from `T = 8`.
- **Backward compatibility:** All existing `ctest` and `python -m unittest discover -s tools/bench -p "test_*.py"` tests must remain green.

## Review Focus

1. **MKL vs Panua symbol collision in one process:** Resolving Panua's 17-argument `pardiso` from the wrong DLL crashes or corrupts the stack when MKL is also linked. Pinned in Task 1 (`PanuaAndMklCoexistInSameProcess`, explicit `GetProcAddress`/`dlsym` on `pardiso_d` / `pardisoinit_d`).
2. **Missing or invalid Panua library/license (`-10`/`-11`/`-12`):** Must return `Outcome::Unavailable` (never `Failed` or abort) so CI and unlicensed hosts stay green. Pinned in Task 1 (`PanuaUnavailableWhenLibOrLicenseMissing`).
3. **GPU (`cudss`) exceeding 165 GFLOP/s at `T = 1` while mis-threaded CPU solvers also exceed 165 GFLOP/s:** GPU must pass plausibility while CPU solvers must still be flagged `implausible`. Pinned in Task 2 (`test_gpu_exempt_from_cpu_gflops_cap`).
4. **SuiteSparse matrices with explicit `0.0` entries or constant-pressure nullspaces (`||A[:, z] * 1|| == 0`):** Must be detected algebraically pre-seal and tagged `report_only: true` without running any solver. Pinned in Task 3 (`test_algebraic_singularity_check`).
5. **License-capped Panua PARDISO (< 8 cores):** Running `T = 8` on a 2-core license would silently misreport a 2-core run as 8-core. Pinned in Task 4 (`inventory_environment` license probe) and Task 5/6 pre-flight `check_license.exe` gate.

---

### Task 1: Dynamic Panua PARDISO Adapter (`pardiso.cpp`)

**Files:**
- Modify: `solver/benchmarks/adapters/pardiso.cpp:1-302`
- Test: `solver/tests/test_benchmark_adapters.cpp`

**Interfaces:**
- Consumes: `SolverAdapter` (`solver/benchmarks/adapters/adapter.hpp`), `Control`, `CscLower`, `Outcome`, `AdapterIdentity`, `AdapterStats`
- Produces: `make_pardiso_adapter("panua_pardiso") -> std::unique_ptr<SolverAdapter>` reading `SMF_PANUA_PARDISO_LIB` (fallback `"libpardiso.dll"` on Windows / `"libpardiso.so"` on Linux), calling `pardisoinit_d` / `pardiso_d` (fallback `pardisoinit` / `pardiso` from the loaded handle only)

- [ ] **Step 1: Write failing C++ tests in `solver/tests/test_benchmark_adapters.cpp`**

Add two tests at the end of `solver/tests/test_benchmark_adapters.cpp`:
1. `TEST(BenchmarkAdapters, PanuaUnavailableWhenLibOrLicenseMissing)`: sets `SMF_PANUA_PARDISO_LIB` to a nonexistent path (`"__nonexistent_panua_lib__.dll"`), restores prior env var via RAII guard, constructs `make_solver_adapter("panua_pardiso")`, asserts `ident.solver_id == "panua_pardiso"`, `EXPECT_FALSE(ident.available)`, `EXPECT_EQ(adapter->analyse(I3), Outcome::Unavailable)`, and `adapter->stats().reason` is non-empty.
2. `TEST(BenchmarkAdapters, PanuaAndMklCoexistInSameProcess)`: if `SMF_PANUA_PARDISO_LIB` is unset/empty, `GTEST_SKIP() << "SMF_PANUA_PARDISO_LIB not set"`. Otherwise constructs both `make_solver_adapter("mkl_pardiso")` and `make_solver_adapter("panua_pardiso")` in the same test, asserts both `identity().available == true` (skipping cleanly if Panua returns license-error `Unavailable` with reason containing `"license"`), runs SPD `3x3` (`diag(2, 4, 8)`) and indefinite `2x2` exchange matrix (`make_exchange_2x2_indef()`) through both adapters sequentially, checks `is_qualified_validation(validate(...))` and indefinite inertia `(pos=1, neg=1, zero=0)`, and checks `factor_entries > 0`.

- [ ] **Step 2: Build and run the new test to verify it fails**

Run:
```powershell
cmake --build build --config Release --target test_benchmark_adapters -j
ctest --test-dir build -C Release -R test_benchmark_adapters --output-on-failure
```
Expected: `PanuaUnavailableWhenLibOrLicenseMissing` passes or fails depending on stub reason; when `SMF_PANUA_PARDISO_LIB` points to extracted `libpardiso.dll`, `PanuaAndMklCoexistInSameProcess` FAILS because `make_pardiso_adapter("panua_pardiso")` returns `UnavailablePardisoAdapter`.

- [ ] **Step 3: Implement `PanuaPardisoSolverAdapter` in `solver/benchmarks/adapters/pardiso.cpp`**

Implement `PanuaPardisoSolverAdapter` inside `solver/benchmarks/adapters/pardiso.cpp` (compiled unconditionally on Windows and Linux using `<windows.h>` `LoadLibraryA`/`GetProcAddress`/`FreeLibrary` on `_WIN32` and `<dlfcn.h>` `dlopen`/`dlsym`/`dlclose` on POSIX):
- Typedefs matching Panua's `pardiso.h`:
  - `using PanuaInitFn = void (*)(void *, int *, int *, int *, double *, int *);`
  - `using PanuaRunFn = void (*)(void *, int *, int *, int *, int *, int *, void *, int *, int *, int *, int *, int *, int *, void *, void *, int *, double *);`
- In constructor: load library from `std::getenv("SMF_PANUA_PARDISO_LIB")` (if non-empty) else default `"libpardiso.dll"` (`_WIN32`) / `"libpardiso.so"` (POSIX). Resolve `"pardisoinit_d"` and `"pardiso_d"` first (fallback `"pardisoinit"` / `"pardiso"` strictly from that loaded module handle — never `RTLD_DEFAULT`). If loading fails, record `available_ = false` and `unavail_reason_`.
- Run a one-time license probe (`pardisoinit_d` with `mtype = -2`, `solver = 0`) in constructor: if `error` is `-10` (no `panua.lic`), `-11` (expired), or `-12` (wrong user/host), set `available_ = false` and `unavail_reason_` with the exact Panua license error message so `identity().available` is `false` and `analyse/factor/solve` return `Outcome::Unavailable`.
- `configure(const Control &ctrl, const std::string &track)`: store `indefinite_ = (ctrl.matrix_type == MatrixType::RealSymmetricIndefinite)`, `track_ = track`, `threads_ = std::max(1, ctrl.num_threads)`.
- `analyse(const CscLower &A)`: `release_handle()`, call `pardisoinit_d(pt_, &session_mtype_, &solver_, iparm_, dparm_, &error)` with `session_mtype_ = indefinite_ ? -2 : 2`, `solver_ = 0`. Set `iparm_[2] = threads_` and `iparm_[5] = 0`. Convert `A.col_ptr` and `A.row_idx` to 1-based `int` vectors `ia_` and `ja_` (symmetric lower-CSC == upper-CSR). Run `phase = 11`, time `stats_.analyse_ms`, capture `stats_.factor_entries = iparm_[17] > 0 ? iparm_[17] : 0` and `stats_.flops` (confirm Panua's `iparm_[18]` unit — MFLOPS vs GFLOPS — against `libpardiso.dll` and document in comment; scale to raw flops only if positive).
- `factor(std::span<const double> values)`: copy `values` into `a_`, run `phase = 22`, time `stats_.factor_ms`, record `inertia_pos = iparm_[21]`, `inertia_neg = iparm_[22]`, `inertia_zero = max(0, n_ - pos - neg)` for indefinite (`n_, 0, 0` for SPD), map error codes (`0 -> Success`, `-1 -> InvalidInput`, `-2/-9/-10 -> OutOfMemory`, `-4 -> Singular`, else `Singular`/`Failed`).
- `solve(std::span<double> b_and_x, int nrhs)`: allocate `x_(n * nrhs)`, run `phase = 33`, time `stats_.solve_ms`, copy `x_` back into `b_and_x` on `error == 0`.
- `release_handle()`: call `phase = -1` when `n_ > 0`, zero `pt_`, set `n_ = 0`; free DLL handle in destructor.
- Update `make_pardiso_adapter(const std::string &variant)` to return `std::make_unique<PanuaPardisoSolverAdapter>()` when `variant == "panua_pardiso"`.

- [ ] **Step 4: Build and run adapter unit tests**

Run:
```powershell
cmake --build build --config Release --target test_benchmark_adapters bench_replay -j
ctest --test-dir build -C Release -R test_benchmark_adapters --output-on-failure
```
Expected: PASS (both with `SMF_PANUA_PARDISO_LIB` unset and when pointed at extracted `libpardiso.dll` + `panua.lic`).

- [ ] **Step 5: Commit Task 1**

```bash
git add solver/benchmarks/adapters/pardiso.cpp solver/tests/test_benchmark_adapters.cpp
git commit -m "feat(bench): implement dynamic Panua PARDISO adapter"
```

---

### Task 2: Scorer Updates for Panua PARDISO, GPU Exemption & Dual Headline Summary (`score.py`)

**Files:**
- Modify: `tools/bench/score.py:53-80,160-185,373-697`
- Test: `tools/bench/test_score.py`

**Interfaces:**
- Consumes: JSONL records with `solver_id` (`"panua_pardiso"`, `"cudss"`, etc.), manifest `class` (`"public_indef_kkt"`, `"public_indef_pde"`, `"public_spd"`), manifest `split` (`"sealed_public"`)
- Produces:
  - `GPU_SOLVERS: frozenset[str] = frozenset({"cudss"})`
  - `INDEF_COMPETITORS` including `"panua_pardiso"`
  - `SPD_COMPETITORS` including `"panua_pardiso"`
  - `CLASS_ORDER` including `"public_indef_kkt"`, `"public_indef_pde"`, `"public_spd"`
  - `score(..., include_gpu: bool = False)` and `render(result, split, include_gpu: bool = False)` + CLI `--include-gpu` and `--split sealed_public`

- [ ] **Step 1: Write failing unit tests in `tools/bench/test_score.py`**

Add tests to `tools/bench/test_score.py`:
1. `test_panua_pardiso_in_competitor_sets`: asserts `"panua_pardiso"` is in `score.competitor_set("hard_indef")`, `score.competitor_set("public_indef_kkt")`, `score.competitor_set("public_indef_pde")`, and `score.competitor_set("public_spd")`; scores a synthetic manifest where `panua_pardiso` is the fastest accurate rival and verifies `best_solver_counts == {"panua_pardiso": 1}`.
2. `test_gpu_exempt_from_cpu_gflops_cap`: builds two `T=1` records both with `flops=5e11`, `factor_ms=1000.0` (`500 GF/s > 165 GF/s` cap): one for `solver_id="cudss"` and one for `solver_id="ssids"`. Asserts `score.is_implausible(cudss_rec) is False` and `score.is_implausible(ssids_rec) is True`.
3. `test_public_classes_and_sealed_split`: scores a manifest with `split="sealed_public"` and classes `public_indef_kkt`, `public_indef_pde`, `public_spd` using `split="sealed_public"` and verifies all three classes appear in `CLASS_ORDER` order.
4. `test_include_gpu_dual_summary_and_rollup`: scores a synthetic `sealed_public` board with `smf`, `panua_pardiso`, `ma97`, and `cudss` across `public_indef_kkt` (`recommended` + `ipopt`) and `public_spd` (`recommended`) at `T=1`, where `cudss` is faster on SPD and `Inaccurate` on indef. Verifies:
   - with `include_gpu=False` (default): `cudss` is NOT in `ratio["best_solver_counts"]` (backward-compatible).
   - with `include_gpu=True`: `render(res, "sealed_public", include_gpu=True)` contains a cross-class headline roll-up section per `T` showing: (a) **CPU-only** geomean & solved count, (b) **GPU head-to-head (`smf` vs `cudss`)** geomean & solved count on common accurate cases, (c) **All rivals incl. GPU** geomean, and (d) per-class breakdown lines.

- [ ] **Step 2: Run unit tests to verify they fail**

Run:
```powershell
python -m unittest discover -s tools/bench -p "test_score.py" -v
```
Expected: FAIL on the 4 newly added tests.

- [ ] **Step 3: Implement updates in `tools/bench/score.py`**

Modify `tools/bench/score.py`:
- Add `GPU_SOLVERS = frozenset({"cudss"})`.
- Update `CLASS_ORDER = ["small_kkt", "large_kkt", "hard_indef", "spd", "public_indef_kkt", "public_indef_pde", "public_spd"]`.
- Add `"panua_pardiso"` to `INDEF_COMPETITORS` and `SPD_COMPETITORS`.
- Update `competitor_set(cls: str)` so `cls in ("spd", "public_spd")` returns `SPD_COMPETITORS` else `INDEF_COMPETITORS`.
- Update `is_implausible(rec, max_gflops_per_thread)`: return `False` immediately if `rec.get("solver_id") in GPU_SOLVERS`.
- Add `"sealed_public"` to `--split` choices in `main()`.
- Add helper `compute_gpu_and_rollup(result, shift_ms, subject)` (or store per-table `gpu_h2h` + `ratio_with_gpu` inside `score()` so `render(..., include_gpu=True)` stays pure and testable):
  - Per `(cls, track, threads)` cell: compute `gpu_h2h` (`smf` vs `cudss` shifted geomean + solved counts on common accurate cohort) and `ratio_with_gpu` (`smf` vs fastest accurate solver in `eligible + GPU_SOLVERS`).
  - In `render(result, split, include_gpu: bool = False)`: when `include_gpu=True`, prepend a `## Headline summary (per thread count)` section listing per `T`: CPU-only overall shifted geomean (per case-track), per-class CPU-only ratios, `smf` vs `cudss` head-to-head ratio & accuracy counts, and All-rivals-incl-GPU ratio.
- Wire `--include-gpu` flag in `main()`.

- [ ] **Step 4: Run full `tools/bench` test suite to verify pass and zero regressions**

Run:
```powershell
python -m unittest discover -s tools/bench -p "test_score.py" -v
```
Expected: ALL PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/bench/score.py tools/bench/test_score.py
git commit -m "feat(bench): score panua_pardiso, exempt GPU GFLOP/s cap, add dual CPU/GPU rollup"
```

---

### Task 3: Sealed Public SuiteSparse Catalog & Fetcher (`fetch_public_suitesparse.py`)

**Files:**
- Create: `tools/bench/fetch_public_suitesparse.py`
- Test: `tools/bench/test_fetch_public_suitesparse.py`

**Interfaces:**
- Consumes: `benchmarks/manifests/fronttier.json` (read-only exclusion source), `tools/bench/fetch_suitesparse.py` (`MIRRORS`, `download`, `read_header`, `sha256_file`)
- Produces:
  - `fetch_public_suitesparse.py` CLI (`--fetch`, `--ssstats`, `--fronttier`, `--out`, `--manifest`, `--budget-mb`)
  - Pure testable functions: `fronttier_exclusion_stems(manifest) -> tuple[set[str], set[str]]`, `parse_ssstats(csv_text) -> list[dict]`, `select_candidates(entries, excluded_names, excluded_stems, targets=...) -> dict[str, list[dict]]`, `check_algebraic_singularity(mtx_path) -> tuple[bool, str]`
  - Output manifest `benchmarks/manifests/public_suitesparse.json` with `schema_version: 1`, `suite: "public_suitesparse"`, `split: "sealed_public"`, `solvers: ["smf", "ma57", "ma97", "mkl_pardiso", "panua_pardiso", "mumps", "cholmod", "ssids", "cudss"]`, and 95 case-tracks (`public_indef_kkt`: 20×2, `public_indef_pde`: 15×2, `public_spd`: 25×1).

- [ ] **Step 1: Write failing unit tests in `tools/bench/test_fetch_public_suitesparse.py`**

Create `tools/bench/test_fetch_public_suitesparse.py` covering:
1. `test_series_sibling_exclusion`: loads `fronttier.json`, verifies `fronttier_exclusion_stems()` excludes exact names (`af_shell7`, `c-big`, `cont-201`, `TSOPF_FS_b300`, `ncvxqp9`), excludes series siblings (`af_shell8`, `cont-200`, `TSOPF_FS_b162_c2`, `nd12k`, `thermal3`), and KEEPS distinct `Schenk_IBMNA` problems (`c-55`, `c-73`).
2. `test_parse_and_select_caps_and_order`: passes a synthetic `ssstats.csv` fixture with >30 entries across multiple groups, graph/combinatorial entries, binary/unsymmetric entries, and out-of-range sizes (`n < 20000`, `n > 600000`, `nnz > 30000000`). Verifies:
   - non-numerical graph/combinatorial, binary, unsymmetric, and out-of-range entries are dropped,
   - per-group caps (`10` for `public_indef_kkt`, `3` for `public_indef_pde`, `3` for `public_spd`) are strictly respected,
   - selection is deterministic and respects target counts.
3. `test_algebraic_singularity_check`: writes 3 temporary Matrix Market files: (a) nonsingular `3x3` saddle-point matrix, (b) matrix with an explicit `0.0` entry creating an all-zero row after dropping zeros, (c) unpinned saddle-point matrix with zero-diagonal rows `z` satisfying `||A[:, z] * 1|| == 0`. Verifies `check_algebraic_singularity` returns `(False, "")` for (a), and `(True, reason)` for (b) and (c).
4. `test_manifest_case_tracks_and_report_only`: verifies indefinite matrices emit both `ipopt` and `recommended` tracks (`split="sealed_public"`), SPD matrices emit `recommended` only, and a matrix flagged by `check_algebraic_singularity` gets `"report_only": True` on both tracks.

- [ ] **Step 2: Run unit tests to verify they fail**

Run:
```powershell
python -m unittest discover -s tools/bench -p "test_fetch_public_suitesparse.py" -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'fetch_public_suitesparse'`.

- [ ] **Step 3: Implement `tools/bench/fetch_public_suitesparse.py`**

Implement the module with:
- `SSSTATS_URL = "https://www.cise.ufl.edu/research/sparse/matrices/ssstats.csv"` (cacheable via `--ssstats <local_csv>`).
- `SERIES_PREFIXES = ("TSOPF_FS_", "af_shell", "thermal", "ncvxqp", "cvxqp", "cont-", "boyd", "aug3d", "cfd", "Dubcova", "apache", "stokes", "mario", "darcy", "bratu", "helm", "copter", "dawson", "blockqp")` + `nd<digits>k` rule; exact names from `fronttier.json` always excluded (so `c-big` is excluded while `c-55`..`c-73` remain eligible).
- `EXCLUDED_KINDS = ("graph", "combinatorial", "directed", "undirected", "weighted", "bipartite", "duplicate")`.
- `KKT_KINDS = ("optim", "linear programming", "power network", "economic", "circuit")`.
- `TARGETS = {"public_indef_kkt": (20, 10), "public_indef_pde": (15, 3), "public_spd": (25, 3)}` (`(target_count, per_group_cap)`).
- Deterministic selection: within each group, sort by `(nrows, name)` and pick up to `per_group_cap` evenly spaced in `log(nrows)`; across groups in a class, sort groups alphabetically and take round-robin up to `target_count`.
- With `--fetch`: iterate through the deterministic ordered candidate list (holding extras as deterministic replacements if a download/header check fails or exceeds `--budget-mb 2500`), download via `fetch_suitesparse.download()`, extract with `tarfile` `"data"` filter, verify `symmetric` real coordinate header, compute `sha256_file()`, run `check_algebraic_singularity()` using `scipy.io.mmread` (or streaming row-sum/zero-diag check in pure numpy so it's fast even at `nnz=15M`!), and write `benchmarks/manifests/public_suitesparse.json` with a top-level `selection_policy` metadata block documenting the exact filter & `ssstats.csv` date.

- [ ] **Step 4: Run unit tests to verify they pass**

Run:
```powershell
python -m unittest discover -s tools/bench -p "test_fetch_public_suitesparse.py" -v
```
Expected: ALL PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/bench/fetch_public_suitesparse.py tools/bench/test_fetch_public_suitesparse.py
git commit -m "feat(bench): add pre-registered SuiteSparse public corpus fetcher"
```

---

### Task 4: WSL2 Linux `smf` OS-Parity Runner & Provenance Inventory (`run_wsl.py`, `run.py`)

**Files:**
- Modify: `tools/bench/run_wsl.py:1-200`
- Modify: `tools/bench/run.py:1-100` (`inventory_environment`)
- Test: `tools/bench/test_run_wsl.py`

**Interfaces:**
- Consumes: `tools/bench/run_wsl.py` (`to_wsl_path`, `wsl_command`), `tools/bench/run.py` (`inventory_environment`)
- Produces:
  - `run_wsl.py`: `smf_driver_path(distro) -> str` (reads `SMF_WSL_SMF_BINARY`, default `~/opt_solver_wsl/build/benchmarks/bench_replay`), `wsl_smf_command(args, threads, binary, distro, timeout_s) -> list[str]` (sets `OMP_NUM_THREADS=T`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, no SSIDS-only `OMP_CANCELLATION`), and CLI `--solver ssids|smf` in `run_wsl.py` so `python tools/bench/run_wsl.py --solver smf --manifest ...` runs Linux OpenBLAS `smf` via the exact same WSL2 runner!
  - `run.py`: `inventory_environment()` records `panua_pardiso_lib`, `panua_lic_path`, `panua_license_check` (stdout of `check_license.exe` if present next to `libpardiso.dll`), and `gpu` (`nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader` if available).

- [ ] **Step 1: Write failing unit tests in `tools/bench/test_run_wsl.py`**

Add tests to `tools/bench/test_run_wsl.py`:
1. `test_wsl_smf_command_env_and_paths`: calls `run_wsl.wsl_smf_command(["--solver", "smf", "--mtx", r"C:\data\a.mtx", "--JsonlOut", r"C:\out\r.jsonl"], threads=8, binary="/home/u/build/bench_replay", distro="Ubuntu", timeout_s=300)` and asserts:
   - paths translated to `/mnt/c/data/a.mtx` and `/mnt/c/out/r.jsonl`,
   - env contains `OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
   - env does NOT inject `OMP_CANCELLATION=TRUE`,
   - invokes `/home/u/build/bench_replay`.
2. `test_inventory_environment_includes_panua_and_gpu_keys`: calls `run.inventory_environment(pathlib.Path("build"))` and verifies keys `"panua_pardiso_lib"`, `"panua_lic_path"`, `"gpu"` exist in the returned dict (never crashing if Panua or `nvidia-smi` is absent).

- [ ] **Step 2: Run unit tests to verify they fail**

Run:
```powershell
python -m unittest discover -s tools/bench -p "test_run_wsl.py" -v
```
Expected: FAIL (`AttributeError: module 'run_wsl' has no attribute 'wsl_smf_command'`).

- [ ] **Step 3: Implement WSL2 `smf` dispatch in `tools/bench/run_wsl.py` and provenance fields in `tools/bench/run.py`**

- In `tools/bench/run_wsl.py`: add `smf_driver_path()`, `wsl_smf_command()`, and `--solver` choice (`ssids`, `smf`, default `ssids`) in `run_wsl.py`'s standalone runner (and in `run.py` allow `--runner wsl` with `--solvers smf` via a small dispatch hook or standalone `run_wsl.py --solver smf` writing directly into its own WSL2 parity output directory so Windows `smf` and Linux `smf` records never collide in the same JSONL!).
- In `tools/bench/run.py`: extend `inventory_environment()` to record `panua_pardiso_lib`, `panua_lic_path`, best-effort `panua_license_check` (if `../bin/check_license.exe` exists next to `SMF_PANUA_PARDISO_LIB`), and best-effort `gpu` from `nvidia-smi`.

- [ ] **Step 4: Run unit tests to verify they pass**

Run:
```powershell
python -m unittest discover -s tools/bench -p "test_run_wsl.py" -v
python -m unittest discover -s tools/bench -p "test_*.py"
```
Expected: ALL PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add tools/bench/run_wsl.py tools/bench/run.py tools/bench/test_run_wsl.py
git commit -m "feat(bench): add WSL2 Linux smf parity runner and Panua/GPU env inventory"
```

---

### Task 5: Pre-Flight Rival Verification & Sealed Manifest Pre-Registration Commit

**Files:**
- Create (gitignored): `build-deps/panua-pardiso/` (extracted from `C:\Users\daniel\Downloads\panua-pardiso-20240630-win.zip`)
- Create (gitignored): `benchmarks/fixtures/public_suitesparse/` (downloaded `.mtx` files)
- Create (committed): `benchmarks/manifests/public_suitesparse.json`

**Interfaces:**
- Consumes: `build/benchmarks/Release/bench_replay.exe` (Task 1), `tools/bench/fetch_public_suitesparse.py` (Task 3), `tools/bench/run.py` (Task 4), user's `%USERPROFILE%\panua.lic`
- Produces: Committed, frozen `benchmarks/manifests/public_suitesparse.json` + verified Panua license core cap + green smoke run across all 9 solvers on `smoke.json` (zero exposure to any new matrix)

- [ ] **Step 1: Extract Panua PARDISO archive to gitignored `build-deps/panua-pardiso/` and verify license**

Extract `C:\Users\daniel\Downloads\panua-pardiso-20240630-win.zip` into `build-deps/panua-pardiso/` (confirm `git status` ignores `build-deps/`).
Ask Daniel to place his Panua license key in `%USERPROFILE%\panua.lic` (if not already there), then run:
```powershell
& build-deps\panua-pardiso\panua-pardiso-20240630-win\bin\check_license.exe
```
Record the exact license tier and maximum allowed cores `K` from its output.

- [ ] **Step 2: Smoke-test every rival on `benchmarks/manifests/smoke.json` (NEVER on any new matrix)**

Set `SMF_PANUA_PARDISO_LIB` to `build-deps\panua-pardiso\panua-pardiso-20240630-win\lib\libpardiso.dll` (plus `SMF_HSL_LIBRARY` and `PATH` for CoinHSL/MKL/cuDSS DLLs as needed) and run:
```powershell
ctest --test-dir build -C Release -R test_benchmark_adapters --output-on-failure
python tools/bench/run.py --manifest benchmarks/manifests/smoke.json --output build/smoke_all_rivals --trials 1 --warmups 0 --allow-short-trials --threads 1 --solvers smf,ma57,ma97,mkl_pardiso,panua_pardiso,mumps,cholmod,cudss
```
Verify every solver reports `Success` (not `Unavailable` or crash) on `smoke.json` and check WSL2 `ssids` + Linux OpenBLAS `smf` on `smoke.json` via `tools/bench/run_wsl.py`.

- [ ] **Step 3: Fetch the sealed public SuiteSparse corpus and build `public_suitesparse.json`**

Run:
```powershell
python tools/bench/fetch_public_suitesparse.py --fetch --budget-mb 2500 --out benchmarks/fixtures/public_suitesparse --manifest benchmarks/manifests/public_suitesparse.json
```
Verify:
- 60 unique matrices → 95 case-tracks (`public_indef_kkt`: 40, `public_indef_pde`: 30, `public_spd`: 25),
- 0 overlap (exact or series sibling) with `fronttier.json`,
- every entry has a non-empty `matrix_sha256`,
- any algebraically singular matrices are logged and marked `"report_only": true`.

- [ ] **Step 4: Commit the sealed pre-registration manifest (BEFORE running `smf` on it)**

Verify `git status` shows ONLY `benchmarks/manifests/public_suitesparse.json` (no `.mtx`, no `panua.lic`, no `libpardiso.dll`), then commit:
```bash
git add benchmarks/manifests/public_suitesparse.json
git commit -m "bench: pre-register sealed public SuiteSparse manifest (60 matrices, 95 tracks)"
```

---

### Task 6: Run the Blind Full-Rival Board, Post-Seal Inertia Reference & Unfiltered Report

**Files:**
- Create (gitignored raw data): `benchmarks/results/public_suitesparse_20261122/` (Windows native + WSL2 SSIDS + WSL2 Linux `smf` JSONLs and `environment.json`)
- Create (committed): `benchmarks/manifests/public_suitesparse_inertia.json`
- Create (committed): `docs/performance/2026-11-22-public-suitesparse-board.md`

**Interfaces:**
- Consumes: Sealed commit of `benchmarks/manifests/public_suitesparse.json` (Task 5), `tools/bench/run.py`, `tools/bench/run_wsl.py`, `tools/bench/inertia_ref.py`, `tools/bench/score.py --include-gpu`
- Produces: Unfiltered public benchmark write-up comparing `smf` against MA57, MA97, MKL PARDISO, Panua PARDISO, MUMPS, CHOLMOD, SSIDS (WSL2), and cuDSS (RTX 5080)

- [ ] **Step 1: Run the Windows native board at `T = 1` and `T = 8` (+ `T = K` for Panua if license-capped at `K < 8`)**

Ensure no heavy background jobs are running.
- If Panua's license supports `T >= 8`:
  ```powershell
  python tools/bench/run.py --manifest benchmarks/manifests/public_suitesparse.json --output benchmarks/results/public_suitesparse_20261122/win --trials 3 --warmups 1 --allow-short-trials --timeout 300 --threads 1,8 --serial-solvers ma57 --solvers smf,ma57,ma97,mkl_pardiso,panua_pardiso,mumps,cholmod,cudss
  ```
- If Panua's license caps at `K < 8` (e.g. `K = 2`): pass `--serial-solvers ma57,panua_pardiso` on the `1,8` run, then run a dedicated matched-thread `T = K` comparison (`--threads K --solvers smf,panua_pardiso`) into `benchmarks/results/public_suitesparse_20261122/panua_tK/`.

- [ ] **Step 2: Run the WSL2 `ssids` + Linux OpenBLAS `smf` OS-parity board on the same host**

Run:
```powershell
python tools/bench/run_wsl.py --solver ssids --manifest benchmarks/manifests/public_suitesparse.json --output benchmarks/results/public_suitesparse_20261122/wsl_ssids --trials 3 --warmups 1 --allow-short-trials --timeout 300 --threads 1,8
python tools/bench/run_wsl.py --solver smf --manifest benchmarks/manifests/public_suitesparse.json --output benchmarks/results/public_suitesparse_20261122/wsl_smf --trials 3 --warmups 1 --allow-short-trials --timeout 300 --threads 1,8
```

- [ ] **Step 3: Build post-seal inertia reference (`public_suitesparse_inertia.json`)**

Run (strictly *after* Steps 1–2):
```powershell
python tools/bench/inertia_ref.py --manifest benchmarks/manifests/public_suitesparse.json --binary build/benchmarks/Release/bench_replay.exe --output benchmarks/manifests/public_suitesparse_inertia.json --threads 1 --timeout 120
```
(`inertia_ref.py` already handles per-subprocess `--timeout` and writes non-ok statuses that `score.py` skips.)

- [ ] **Step 4: Score all combinations and write the unfiltered report**

Run `score.py` to produce:
1. Main board (`win/records.jsonl` + `wsl_ssids/records.jsonl`) scored with `--split sealed_public --include-gpu --inertia-ref benchmarks/manifests/public_suitesparse_inertia.json`,
2. WSL2 OS-parity table (`wsl_smf/records.jsonl` + `wsl_ssids/records.jsonl`) scored with `--split sealed_public --inertia-ref benchmarks/manifests/public_suitesparse_inertia.json`,
3. If Panua was license-capped at `K < 8`: dedicated `T = K` `smf` vs `panua_pardiso` table.

Write `docs/performance/2026-11-22-public-suitesparse-board.md` containing:
- Pre-registration commit SHA, exact hardware/OS/compiler/library versions & Panua license cap,
- CPU-only headline ratios (`1T`, `8T`, and `K`T if applicable) + per-class breakdown (`public_indef_kkt`, `public_indef_pde`, `public_spd`),
- GPU head-to-head (`smf` vs `cudss` on RTX 5080) + All-rivals-incl-GPU summary,
- WSL2 same-OS/same-BLAS parity table (`smf` Linux OpenBLAS vs `ssids`),
- Full accuracy, inertia, OOM/timeout, and worst-losses tables (unfiltered — every single loss listed honestly).

- [ ] **Step 5: Commit inertia reference and published report**

```bash
git add benchmarks/manifests/public_suitesparse_inertia.json docs/performance/2026-11-22-public-suitesparse-board.md
git commit -m "docs(perf): publish blind sealed SuiteSparse benchmark report"
```
