# Sealed Public SuiteSparse Benchmark & Full-Rival Comparison — Design

**Date:** 2026-11-22
**Project:** Layer 1 of a two-layer public proof of `smf` solver leadership (Layer 2: public end-to-end IPOPT on Mittelmann AMPL-NLP, expanded PGLib-OPF, COPS 3.0, and CUTEst).
**Status:** Reviewed by fresh builder-check — ready for user approval.

---

## 1. Intent and Why

`smf` currently beats MA57, MA97, MKL PARDISO, SSIDS, and MUMPS on the 144-track `fronttier.json` board (0.67 at 1T, 0.50 at 8T). However, that board cannot stand alone as an external "#1" proof for two reasons:

1. **Contamination & custom cases:** 24% of `fronttier.json` consists of our own Pyomo/synthetic matrices (`optim20` + `ipopt_kkt`), and `smf` has been heavily tuned across its entire matrix list (including the old 30% held-out split).
2. **Missing top rivals on the same board:** Neither **Panua PARDISO** (currently an unimplemented stub in `solver/benchmarks/adapters/pardiso.cpp`) nor **NVIDIA cuDSS** (excluded from the scored competitor set in `tools/bench/score.py` and absent from the 12900K board) is raced head-to-head on a single machine alongside MA57, MA97, MKL PARDISO, MUMPS, SSIDS, and CHOLMOD.

This design builds a **sealed, 100% public, zero-overlap SuiteSparse matrix benchmark** committed *before* `smf` ever runs on it, plus full **Panua PARDISO** and **cuDSS** integration on one machine (Ryzen 9 9950X3D + NVIDIA RTX 5080), so we get an honest, uncontaminated verdict — first for ourselves, then publishable without cherry-picking caveats.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
1. **100% public, strictly disjoint matrix corpus:** A new manifest `benchmarks/manifests/public_suitesparse.json` populated verbatim from the SuiteSparse Matrix Collection with **zero overlap** with the 45 SuiteSparse matrices already in `fronttier.json`, selected by deterministic, pre-registered criteria over SuiteSparse's official index (`ssstats.csv`).
2. **Full rival roster on one host:** Race `smf` head-to-head against **MA57**, **MA97**, **MKL PARDISO**, **Panua PARDISO**, **MUMPS**, **CHOLMOD** (SPD), **NVIDIA cuDSS** (GPU), and **SPRAL SSIDS** (WSL2 on the same physical Ryzen 9 9950X3D host, clearly labeled as WSL2/Linux alongside a Linux `smf` OpenBLAS/MKL row for OS parity).
3. **Real Panua PARDISO adapter:** Implement `panua_pardiso` in `solver/benchmarks/adapters/pardiso.cpp` via dynamic library loading (`libpardiso.dll` / `libpardiso.so`) with zero symbol collision against Intel MKL PARDISO, using Panua's official `pardisoinit_d` / `pardiso_d` C ABI and recording license type and thread cap.
4. **Fair cuDSS & Panua scoring in `score.py`:** Add `panua_pardiso` to the scored CPU competitor sets, exempt GPU solvers (`cudss`) from the CPU per-thread GFLOP/s plausibility cap, and report both **CPU-only** and **CPU + GPU** headline summaries (speedup ratio *and* accuracy count side by side).
5. **Pre-registered sealing & reproducibility:** Commit the catalog script, manifest (with matrix SHA-256 hashes and pre-run structural singularity tags), adapter, and scorer changes **before** running `smf` on a single matrix from the new set.

### Non-Goals
- **No tuning `smf` on the new set.** If the blind run exposes weaknesses, any follow-up tuning is a separate project and must be reported as "post-tuning", never "blind".
- **No edits to `fronttier.json`.** The existing 144-track engineering board remains untouched.
- **No end-to-end IPOPT runs here.** Mittelmann + PGLib-OPF + COPS 3.0 + CUTEst are Project 2.

### Anti-Goals (failure modes to actively prevent)
- Dropping or editing matrices after seeing any solver's results.
- Comparing a license-capped Panua PARDISO run (e.g., 2-core academic trial) as if it were an 8-thread run.
- Crippling any rival with non-recommended defaults (e.g. 1×1-only pivoting or disabled matching on indefinite KKT matrices).
- Committing licensed third-party binaries or keys (`panua.lic`, `libpardiso.dll`, `libcoinhsl.dll`) to git.
- Breaking any existing `ctest` or `tools/bench/test_*.py` tests.

---

## 3. Approaches Considered

1. **Sealed Public SuiteSparse Matrix Benchmark only:** Fast, tests raw direct-solver dominance on unseen public matrices, reuses `bench_replay`/`run.py`/`score.py`, but doesn't test full nonlinear optimizer loops.
2. **Public End-to-End IPOPT Benchmark only:** Strongest proof for optimization users (CUTEst, Mittelmann, PGLib-OPF, COPS 3.0), but heavier toolchain (Linux/gfortran/pycutest), slower, and nonlinear iteration divergence complicates pure linear-solver attribution.
3. **Two-Layer Public Proof — Sealed SuiteSparse first, then End-to-End IPOPT *(Picked)*:** Closes both lines of skepticism in order. Layer 1 (this spec) delivers a fast, uncontaminated matrix-level verdict across every major CPU and GPU direct solver before investing in Layer 2's full IPOPT/CUTEst suite.

---

## 4. Design

### 4.1 Pre-Registered Matrix Selection (`tools/bench/fetch_public_suitesparse.py`)

Instead of hand-picking matrices or whitelisting groups, a new reproducible script `tools/bench/fetch_public_suitesparse.py` queries SuiteSparse's official CSV index (`https://www.cise.ufl.edu/research/sparse/matrices/ssstats.csv`, verified HTTP 200 with 2,757 entries; Note: `sparse.tamu.edu/files/ssstats.csv` returns HTTP 403) and applies fixed, objective, deterministic rules **before any solver is run**:

- **Hard eligibility filter:**
  - `isReal == 1`, `isBinary == 0`, `nrows == ncols`, `numerical_symmetry == 1.0`.
  - Exclude non-numerical graph/combinatorial kinds (`kind` containing `graph`, `combinatorial`, `directed`, `undirected`, `weighted`, `bipartite`, or `duplicate`).
  - **Strict generator/series-sibling disjointness from `fronttier.json` *(builder/author call)*:** Exclude both exact matrix names AND same-generator series siblings of the 45 SuiteSparse matrices in `fronttier.json` (`TSOPF_FS_*`, `af_shell*`, `thermal*`, `ncvxqp*`, `cvxqp*`, `cont-*`, `boyd*`, `aug3d*`, `nd*k`, `cfd*`, `Dubcova*`, `apache*`, `stokes*`, `mario*`, `darcy*`, `bratu*`, `helm*`, `copter*`, `dawson*`, `blockqp*`) so structural clones (e.g., `af_shell8`, `cont-200`) never contaminate the "unseen" claim. Note: `Schenk_IBMNA`'s `c-*` series (`c-55`..`c-73`, distinct optimization problems) is kept, excluding only `c-big` itself.
  - Size bounds (chosen so every matrix is nontrivial yet fits comfortably in 16 GB GPU VRAM for cuDSS and finishes within the 300 s per-solve timeout):
    - $20{,}000 \le n \le 600{,}000$
    - full symmetric `nnz` in `ssstats.csv`: $\text{nnz} \le 30{,}000{,}000$ (lower triangle $\le \sim 15\text{M}$).
  - Total download budget cap: 2.5 GB compressed across the entire corpus.
- **Deterministic class partitioning & per-group caps (verified against `ssstats.csv`):**
  Running the filter above against `ssstats.csv` yields **42** eligible `public_indef_kkt`, **46** eligible `public_indef_pde`, and **52** eligible `public_spd` matrices — zero shortfall in any class:
  1. **`public_indef_kkt` (target 20 matrices, indefinite, 2 tracks each: `recommended` + `ipopt`):**
     - All eligible unseen real symmetric indefinite (`isPosDef == 0`) matrices whose `kind` matches optimization / linear programming / power network / economic / circuit (currently spans `Schenk_IBMNA`, `GHS_indef`, and `IPSO`).
     - Cap 10 per group (selecting evenly by `log(n)` within each over-cap group), then take 20 evenly by `(group round-robin, log(n))`.
  2. **`public_indef_pde` (target 15 matrices, indefinite, 2 tracks each: `recommended` + `ipopt`):**
     - All other eligible unseen real symmetric indefinite (`isPosDef == 0`) numerical matrices across *all* SuiteSparse groups (currently spans 11 groups: `PARSEC`, `GHS_indef`, `Boeing`, `Oberwolfach`, `Koutsovasilis`, `TKK`, `Cunningham`, `Lin`, `Dziekonski`, `Williams`, `Bodendiek`).
     - Cap 3 per group (evenly by `log(n)`), then select 15 deterministically across groups (`group round-robin, log(n)`).
  3. **`public_spd` (target 25 matrices, SPD, 1 track each: `recommended`):**
     - All eligible unseen real SPD (`isPosDef == 1`) numerical matrices across *all* SuiteSparse groups (currently spans 18 groups: `GHS_psdef`, `DNVS`, `Schenk_AFE`, `Boeing`, `Botonakis`, `Um`, `MaxPlanck`, `Williams`, `Nasa`, `Cunningham`, `AMD`, `Oberwolfach`, `INPRO`, `BenElechi`, `Wissgott`, `Castrillon`, `TKK`, `Mazaheri`).
     - Cap 3 per group (evenly by `log(n)`), then select 25 deterministically across groups (`group round-robin, log(n)`).
  - **Fallback rule:** If any single matrix fails download/header verification, replace it with the next eligible matrix in that class's deterministic order before committing the manifest; if a class ever has fewer eligible matrices than target, keep all eligible matrices without relaxing size/disjointness bounds.
- **Pre-run singularity / inconsistency check (no solver involved):**
  - During manifest construction, after downloading and verifying each `.mtx` header + SHA-256, run a lightweight purely algebraic check (same check used to diagnose `d_pretok`/`turon_m`): empty/zero rows, exact constant-vector nullspace on zero-diagonal rows (`||A[:, z] * 1|| == 0`), or explicit rank-deficiency metadata in SuiteSparse comments.
  - Matrices proven mathematically singular & inconsistent with $b = \mathbf{1}$ are marked `"report_only": true` *in the pre-registered manifest* (never retroactively after a run).
- **Manifest layout (`benchmarks/manifests/public_suitesparse.json`):**
  - Same schema as `fronttier.json` (`schema_version: 1`, `cases` array with `matrix_id`, `sequence_id`, `family_id`, `source`, `mtx_path`, `matrix_sha256`, `n`, `nnz_lower`, `matrix_type`, `mode: "cold"`, `expected_outcome: "Success"`, `class`, `split: "sealed_public"`, `track`, optional `report_only`). Total: 60 unique matrices → **95 case-tracks** (35 indefinite × 2 tracks + 25 SPD × 1 track).
  - Downloaded `.mtx` files live in gitignored `benchmarks/fixtures/public_suitesparse/<name>/<name>.mtx`.

### 4.2 Panua PARDISO Adapter (`solver/benchmarks/adapters/pardiso.cpp`)

Replace `UnavailablePardisoAdapter` for `"panua_pardiso"` with a real `PanuaPardisoSolverAdapter`:

- **Dynamic loading (zero MKL symbol collision):**
  - Never link `libpardiso.lib` at build time (it exports `pardiso` / `pardisoinit` with a different 17-arg/6-arg signature than MKL's 16-arg/3-arg symbols).
  - Load dynamically via `LoadLibraryA` (Windows) / `dlopen` (Linux) using `SMF_PANUA_PARDISO_LIB` (env var path to `libpardiso.dll` / `libpardiso.so`), falling back to `libpardiso.dll` / `libpardiso.so` on the system loader path.
  - Resolve `pardisoinit_d` and `pardiso_d` (fallback: `pardisoinit` and `pardiso` from the loaded module handle). If the library cannot be loaded, gracefully report `Outcome::Unavailable` with a clear reason.
- **Fair Panua configuration:**
  - `mtype = indefinite ? -2 : 2`, `solver = 0` (sparse direct).
  - Call `pardisoinit_d(pt, &mtype, &solver, iparm, dparm, &error)` first so Panua chooses its own recommended defaults (parallel METIS ordering, Bunch–Kaufman pivoting, scaling/matching, perturbation). Map license errors `-10` (no `panua.lic`), `-11` (expired), `-12` (wrong host/user) to `Outcome::Unavailable` with explicit messages.
  - Set `iparm[2] = threads` (matching `OMP_NUM_THREADS`) and `iparm[5] = 0` (write to `x`).
  - Convert CSC lower-triangle `(col_ptr, row_idx)` to 1-based indexing (identical to Panua's own `pardiso_sym.c` example: symmetric lower-CSC equals upper-CSR).
  - On `recommended` track: keep `pardisoinit_d`'s internal refinement default (for parity with MKL PARDISO's adapter, where `iparm[7]` permits up to 2 steps on perturbed pivots; harness external refinement runs only on `ipopt` track for all solvers identically). Record Panua's actual `iparm` values in the adapter comments/docs.
  - On `analyse` (`phase = 11`), `factor` (`phase = 22`), `solve` (`phase = 33`), `release` (`phase = -1`): record `factor_entries` (`iparm[17]`), `flops` (`iparm[18]`, scaled from Panua's MFLOPS/GFLOPS convention — verify units against `pardiso_sym.c` and the Panua manual), inertia (`iparm[21]` pos, `iparm[22]` neg), and `peak_rss_bytes`.
- **License & thread-cap verification:**
  - Before running the board, Daniel places `panua.lic` (from his email) in `%USERPROFILE%\panua.lic` (or sets `PANUA_LIC_PATH`), and we run `check_license.exe` from the extracted Panua archive (extracted outside git, e.g. under `build-deps/panua-pardiso/`, which is gitignored).
  - Record the exact Panua license tier and max supported cores in `environment.json`. If the license caps Panua at fewer than 8 cores (e.g., 2-core academic trial), run Panua at `T = 1` and at its exact licensed cap (e.g., `T = 2`), and never score a 2-core run in an 8-core table.

### 4.3 cuDSS, WSL2 SSIDS Parity & Scoring Updates (`tools/bench/score.py`, `tools/bench/run.py`, `tools/bench/run_wsl.py`)

- **cuDSS (`solver/benchmarks/adapters/cudss_adapter.cpp`):**
  - Already built in `build/` (`SMF_BENCH_CUDSS=ON`, cuDSS 0.8.0.10, RTX 5080 16 GB). Verify `cudss` works cleanly on `bench_replay` before the board starts (including warm context init and OOM handling on large matrices).
  - Keep `cudss` configuration fair: default hybrid/METIS ordering, `CUDSS_MTYPE_SYMMETRIC` for indefinite and `CUDSS_MTYPE_SPD` for SPD, `CUDSS_CONFIG_IR_N_STEPS = 3` on `ipopt` track (already implemented in `cudss_adapter.cpp`).
  - **Thread-matching *(author call)*:** Run `cudss` at **every** board thread count (`T = 1` and `T = 8`, matching `OMP_NUM_THREADS` for cuDSS's host-CPU analysis/ordering) so it participates natively in `score.py`'s `(class, track, threads)` tables without cross-thread joins.
- **WSL2 Linux `ssids` + `smf` OS-parity *(author call)*:**
  - Inside WSL2 on the same Ryzen 9 9950X3D host, build/run `smf` with **OpenBLAS** (matching WSL2 `ssids`'s OpenBLAS backend and `STATUS.md`'s existing WSL2 Linux OpenBLAS workflow) alongside `ssids` via `tools/bench/run_wsl.py` (or a thin Linux `bench_replay` invocation recording `solver_id: "smf"` into a separate WSL2 JSONL).
  - In scoring: `ssids` stays in the main CPU competitor set for comparability with existing boards (labeled as WSL2/OpenBLAS), **and** we also emit a dedicated same-OS / same-BLAS WSL2 parity comparison (`smf` Linux OpenBLAS vs `ssids` Linux OpenBLAS) so no Windows-vs-Linux difference can bias either direction.
- **Scorer (`tools/bench/score.py`):**
  1. Add `"panua_pardiso"` to `INDEF_COMPETITORS` and `SPD_COMPETITORS` (participating in "best CPU rival" whenever present in a board; missing solvers remain cleanly handled).
  2. Add `"public_indef_kkt"`, `"public_indef_pde"`, `"public_spd"` to `CLASS_ORDER` (mapping `"public_spd"` to `SPD_COMPETITORS` in `competitor_set()`), and accept `"sealed_public"` in `--split`.
  3. Exempt GPU solvers (`GPU_SOLVERS = {"cudss"}`) from the CPU per-thread GFLOP/s plausibility gate (`DEFAULT_MAX_GFLOPS_PER_THREAD`), since an RTX 5080 legitimately exceeds 165 GFLOP/s at `threads=1`.
  4. Add an opt-in `--include-gpu` / dual-summary view *(author call: keeps `test_score.py`'s existing reference-only default untouched)* that renders both per-`(class, track, T)` tables and a **cross-class headline roll-up per `T`** showing:
     - **CPU-only leaderboard** (`smf` vs fastest accurate CPU rival at matched `T`: MA57, MA97, MKL PARDISO, Panua PARDISO, MUMPS, SSIDS, CHOLMOD),
     - **GPU head-to-head (`smf` vs `cudss` on RTX 5080)** showing accuracy solved-count and shifted geomean wall-time ratio on their common accurate cohort, plus `all rivals including GPU`,
     - Both per-case-track roll-up (matching `fronttier` conventions) and per-class lines (`public_indef_kkt`, `public_indef_pde`, `public_spd`) side by side so 2-track indefinite vs 1-track SPD weighting is completely transparent.
  5. Keep all existing `tools/bench/test_score.py` tests passing and add unit tests for `panua_pardiso`, `public_*` classes, GPU plausibility exemption, and the dual summary.

### 4.4 Hardware, Execution Protocol, Inertia Verification & Sealing

- **Host:** Single machine — Daniel's **Ryzen 9 9950X3D + NVIDIA GeForce RTX 5080 (16 GB)** — so every single rival (`smf`, MA57, MA97, MKL PARDISO, Panua PARDISO, MUMPS, CHOLMOD, cuDSS, and WSL2 SSIDS + Linux `smf`) runs on the **exact same silicon**.
- **Thread counts & solver participation *(author call)*:**
  - **`T = 1`:** all solvers (`smf`, MA57, MA97, MKL PARDISO, Panua PARDISO, MUMPS, CHOLMOD, SSIDS, cuDSS).
  - **`T = 8`:** `smf`, MA97, MKL PARDISO, MUMPS, **CHOLMOD** (multi-threaded MKL BLAS supernodal SPD — must NOT be omitted at 8T), SSIDS, **cuDSS**, and **Panua PARDISO** if its license permits `T >= 8` (if Panua's license caps threads at `K < 8`, e.g. 2-core academic trial, run Panua + `smf` head-to-head at `T = K` in an explicit `T = K` column and omit Panua from `T = 8`). **MA57** runs at `T = 1` only (passed via `--serial-solvers ma57`), matching `README.md`'s existing serial-MA57 convention.
  - *(Optional stretch: `T = 16` can be added after `1T` + `8T` if desired, using the same runner.)*
- **Measurement hygiene:**
  - 1 warmup + 3 measured trials per `(case, track, solver, T)`; per-case timeout 300 s; no competing heavy workloads on the host during measurement.
  - Record exact binary paths, git SHA, compiler, BLAS/MKL/OpenBLAS version, CoinHSL version, Panua version & license cap, MUMPS version, CHOLMOD version, cuDSS & CUDA driver version, and SSIDS version in `environment.json`.
- **Post-seal inertia reference *(author call)*:**
  - *After* the sealed commit and blind benchmark run (never before, because `inertia_ref.py` invokes `smf`), run `tools/bench/inertia_ref.py` with a per-matrix timeout (e.g. 120 s) to build `benchmarks/manifests/public_suitesparse_inertia.json` so inertia is verified on both `recommended` and `ipopt` tracks; any huge matrix where strict `u = 0.5` spectrum slicing times out is marked unverified and falls back to modal accurate-rival vote on `ipopt`.
- **Sealing order (strict):**
  1. Build & test Panua adapter + cuDSS smoke check + `score.py` updates (using `smoke.json` / `fronttier.json` only — never any matrix from the new set).
  2. Run `fetch_public_suitesparse.py` to generate and freeze `benchmarks/manifests/public_suitesparse.json` with SHA-256 hashes and pre-run algebraic `report_only` tags.
  3. Commit the manifest and tooling to git (`pre-registered sealed commit`).
  4. Run the full board (all rivals + `smf`) once against that exact commit, build the post-seal inertia reference, and publish the unfiltered report in `docs/performance/YYYY-MM-DD-public-suitesparse-board.md`.

---

## 5. What Is Left to the Builder

- Exact SuiteSparse `ssstats.csv` parsing, mirror retry logic, and group-balancing tie-breakers within the pre-registered rules in Section 4.1.
- Exact `build-deps/panua-pardiso/` extraction helper (gitignored) and `SMF_PANUA_PARDISO_LIB` wiring in `tools/bench/run.py`.
- Confirming Panua `iparm[18]` (MFLOPS vs GFLOPS) and `iparm[10]/iparm[12]/iparm[20]` defaults directly against `libpardiso.dll` on `smoke_kkt_4` during adapter unit testing.
- Unit tests (`test_score.py`, `test_fetch_public_suitesparse.py`, and a C++ adapter test for `panua_pardiso` when `SMF_PANUA_PARDISO_LIB` is set).
