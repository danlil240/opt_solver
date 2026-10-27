# Front-Tier smf: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make smf the fastest *and* most robust sparse symmetric indefinite solver for interior-point KKT systems. That means beating or matching the best of MA57, MA97, SPRAL SSIDS, MKL PARDISO and MUMPS in every problem class, with exact inertia and a 1e-10 backward error, proven on a public-style benchmark.

**Architecture:** There are six parts, done roughly in this order.

- **Phase 0** makes the measurement credible: per-step timing, literature test sets, a held-out split, and the missing competitors.
- **Part A** makes analysis cheap: flat storage, constant-factor cuts for small n, and parallel nested dissection.
- **Part B** replaces the dense pivoting kernel with a posteriori threshold pivoting (APTP), the technique that gave SSIDS PARDISO-level speed with threshold-pivoting accuracy. It is surrounded by a matching + METIS ordering, a delay-budgeted static-pivot fallback that is honest about inertia, and FGMRES refinement.
- **Part C** works on the top of the tree: CHOLMOD amalgamation, 2D tile tasks, determinism, small-front kernels, and a faster solve.
- **Part D** targets the class where MA27 and MA57 are the benchmark leaders: small, very sparse IPOPT KKT systems such as PGLib OPF and CUTEst.
- **Phase E** validates the whole thing on held-out data and publishes the results.

**Tech Stack:** C++20 (MSVC 2022 with `/arch:AVX2`, GCC on WSL for SSIDS), SuiteSparse AMD, METIS 5, MKL BLAS, GoogleTest, Python bench tools (`tools/bench/*`), IPOPT 3.14.

**Spec:** There is no design doc. The plan argues from:

- `reports/Front tier sparse LDLT solvers.md`, the research report, with its notes in `research_notes/Front tier sparse LDLT solvers/`
- the measured gaps in `docs/performance/2026-10-25-beating-ma27.md` and `2026-10-25-large-scale-benchmark.md`
- `docs/roadmap.md`

## What the research changed

| Finding (source note) | Consequence for the plan |
|---|---|
| APTP (Duff–Hogg–Lopez 2018/SISC 2020) matched PARDISO on the hard GHS_indef set, which includes cvxqp3, ncvxqp*, bratu3d and cont-201. It was up to about 10x faster than MA97, kept residuals at 1e-11 where PARDISO gave 3.9e-3, and never perturbs a pivot, so inertia stays exact (`pivoting_robustness.md`). | APTP becomes the centrepiece of Part B (Tasks 8–9). Static pivoting is demoted to a counted last resort (Task 10). |
| SSIDS's weak spot is its serial, unblocked fallback for failed columns (`pivoting_robustness.md`). | Task 9 uses a blocked fallback. |
| Our GHS_indef numbers ran at u = 0.01. Hogg–Scott found that a matching ordering with u = 1e-8 brings delays to about 0. HSL's MA97 ordering 8 is METIS on the compressed graph, with 2x2 pairs weighted 2 (`pivoting_robustness.md`, `ordering_analysis.md`). | Phase 0 measures both tracks. Task 7 builds MA97-ordering-8 parity. |
| On CUTEst and PGLib, MA57 and MA27 are the virtual-best solvers. PARDISO and SSIDS lose on small, very sparse KKT systems (Tasseff 2019, `benchmark_evidence.md`). smf's IPOPT factorization is still 1.7–2.2x slower than MA27's on PGLib. | A new Part D. Front-tier means winning there too, not only on large matrices. |
| The serial AMD algorithm hasn't changed since 2015. ParAMD only pays off from 72K rows on 64 threads. Patch-based ND orders leaf pieces with AMD (about 12x faster, 7–11% more fill) (`ordering_analysis.md`). | Small-n analysis is about constant factors (Task 5). Parallel ND uses AMD on its leaves (Task 6). |
| Classical refinement fails after static pivoting on KKT systems, while FGMRES succeeds and is insensitive to τ (Arioli–Duff–Gratton–Pralet 2007, `refinement_mixed_precision.md`). | Task 11 adds FGMRES escalation. |
| No source shows FP32 LDLᵀ meeting 1e-10 on late-iteration NLP KKT systems; the theory predicts failure (`refinement_mixed_precision.md`). cuDSS's wins come from condensed or regularized systems at 1e-6 tolerance (`kernels_parallelism_gpu.md`). | GPU stays out of this plan. cuDSS is only a benchmark baseline (Task 3). |
| Top-of-tree 2D tile tasks are standard in MA87, SSIDS, PaStiX and Tacho. Deterministic mode needs a canonical assembly order and a fixed BLAS thread count per task (`kernels_parallelism_gpu.md`). | Tasks 13–14. |
| BLR (low-rank compression) has no KKT/IPM evidence (`kernels_parallelism_gpu.md`). | Not planned. |

## Global Constraints

- **Accuracy gate:** normwise backward error ≤ 1e-10 and componentwise ≤ 1e-8, after at most the solver's own refinement.
- **Inertia contract:** on the `ipopt` track, the reported inertia is exact. Any perturbed pivot sets `Info::inertia_exact = false` (Task 10), and the IPOPT plugin treats that as a failed factorization. Plugin behaviour otherwise stays as it is.
- **Threads:** `std::thread`, never OpenMP regions (MSVC's OpenMP runtime aborts next to MKL/CHOLMOD).
- **Tuning:** only on the `tuning` split. The `heldout` split runs once per part, for validation.
- **Tests:** every new gtest suite needs its own `add_test` line in `solver/tests/CMakeLists.txt`.
- **Machine load:** builds and runs at below-normal priority with ≤ 2 build jobs. Ask the user before any run longer than 10 minutes or using more than 1 thread (it is their gaming PC).
- **BUILD:** `"/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" --build build --config Release --parallel 2`.
- **CTEST \<Suite\>:** `ctest --test-dir build -C Release -R <Suite> --output-on-failure`.
- **No factor regression:** after every task, smf vs MA27 on optim20 stays ≥ 1.41x geomean (the current value).

## Exit gates (front tier)

"Best competitor" means the fastest *accurate* solver among MA57, MA97, SSIDS, MKL PARDISO and MUMPS that is available after Task 3. All gates are measured on the `tuning` split and again on `heldout`, where no gate may regress by more than 10%.

| Gate | Class (test set) | Speed target | Robustness target |
|---|---|---|---|
| G-small | small sparse IPOPT KKT (PGLib OPF captures, CUTEst KKT, optim20) | shifted geomean, smf / best ≤ 1.0 | 0 failures |
| G-large | large IPOPT KKT replays (the 7 models) and COPS | ≤ 1.0 | 0 failures, IPOPT iteration counts within ±5% of MA57's |
| G-hard | Duff–Hogg–Lopez hard indefinite set (includes GHS_indef cvxqp3, ncvxqp*, bratu3d, cont-*) | ≤ 1.0 vs best accurate | 0 accuracy failures, exact inertia (no static pivots) on the `ipopt` track |
| G-spd | SuiteSparse SPD set (`large.json`) | ≤ 1.0 vs CHOLMOD/PARDISO | 0 failures |
| G-analyse | analyse time on optim20 A1 cases and bmwcra_1/pwtk at 8 threads | ≤ MA27 / ≤ PARDISO | — |

## Review Focus

1. **APTP with every column failing in a tile** (a zero block, or a pure saddle point with an empty (1,1) block). It must delay them all and still produce a valid factor at the parent or root. Covered in Task 8.
2. **Refactorization after a fallback or static pivots.** The next `factor()` starts clean: `static_pivots = 0`, `inertia_exact = true`, no stale backups. Covered in Tasks 9 and 10.
3. **Empty separators and disconnected graphs in parallel ND, and n ≤ 2.** Covered in Task 6.
4. **Determinism.** With `SMF_DETERMINISTIC=ON`, results are bitwise identical across runs and across 1/8/16 threads, including the APTP tile path. Covered in Task 14.
5. **Structurally singular KKT systems** (the matching fails) reaching METIS-compressed ordering, and singular systems (d_pretok, turon_m) reaching FGMRES. These must fall back or report, never loop. Covered in Tasks 7 and 11.

---

## Phase 0: Credible measurement

### Task 1: Per-step analyse timing

**Files:** `solver/include/smf/info.hpp`, `solver/src/symbolic_analysis.cpp`, `solver/benchmarks/bench_replay.cpp`. Test: `solver/tests/test_symbolic.cpp`.

**Interfaces:**
- Produces: `struct AnalysePhaseSeconds { double check, ordering, matching, etree_counts, amalgamate, assembly_tree, keep_build; };` stored in `Info::analyse_phases`. Records gain `"analyse_phases"` in ms.

- [ ] **Step 1: Failing test** `Symbolic.AnalysePhasesSumToTotal`. On a 2D Poisson 60×60 with AMD: every field ≥ 0, `ordering > 0`, and the sum is within [0.8, 1.0] × `analyse_seconds`.
- [ ] **Step 2: CTEST Symbolic.** Expected: compile error.
- [ ] **Step 3: Implement** with `steady_clock` around each numbered step in `symbolic_analysis.cpp`. Matching time goes in `matching`.
- [ ] **Step 4: BUILD, then CTEST Symbolic.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(solver): per-step analyse timing in Info`.

### Task 2: Literature test sets, two tracks, held-out split, scoring

**Files:**
- Create: `benchmarks/manifests/fronttier.json`, `tools/bench/fetch_fronttier.py`, `tools/bench/score.py`, `tools/bench/test_score.py`
- Modify: `tools/bench/compare_table.py` (`--split`, `--phase`)

**Interfaces:**
- Produces: manifest cases tagged with `"split": "tuning"|"heldout"`, `"class": "small_kkt"|"large_kkt"|"hard_indef"|"spd"`, and `"track": "recommended"|"ipopt"`. Every indefinite case appears on both tracks: u = 0.01, and u = 1e-8 with the matching ordering.
- `score.py --records <jsonl> --manifest <json> --split <s>` prints per class:
  - shifted geomean ratio vs the best accurate competitor, with shift 10 ms
  - solved counts and accuracy failures
  - backward error after refinement
  - solve + refine time
  - a performance-profile CSV
  - separate columns for analyse, first factorization, warm refactorization and solve + refine. PARDISO's internal refinement counts in its solve time.
  - 1-thread and tuned-thread results in separate tables
  - an inertia pass/fail column next to every timing on the `ipopt` track

  These are the methods listed in `benchmark_evidence.md` and the report's benchmarking section.

The sets:
- **hard_indef:** the Duff–Hogg–Lopez hard set (24) plus GHS_indef from `large.json`, with the list taken from `benchmark_evidence.md`. A matrix that can't be fetched from SuiteSparse is listed in the manifest's `"missing"` array, not silently dropped.
- **small_kkt:** optim20 KKT cases plus PGLib captures from `ipopt_hard.json`.
- **large_kkt:** the 7 `ipopt_kkt.json` models.
- **spd:** `large.json` SPD.
- **Split:** a roughly 30% held-out share in each class, chosen by a fixed seed (`seed = 20261027`), stored in the manifest.

- [ ] **Step 1: Failing tests** in `test_score.py`, on synthetic records:
  - `test_shifted_geomean_uses_best_accurate`: an inaccurate faster competitor is excluded from "best".
  - `test_split_filter`.
  - `test_failure_counts_by_class`.
  - `test_profile_csv_monotone`.
- [ ] **Step 2: Run** `python -m pytest tools/bench/test_score.py`. Expected: FAIL.
- [ ] **Step 3: Implement** `score.py`, `compare_table.py` flags, `fetch_fronttier.py` (reuse `fetch_suitesparse.py`'s sha256 logic), and the manifest.
- [ ] **Step 4: Run the tests.** Expected: PASS. Then fetch the new matrices into `benchmarks/fixtures/large/` (gitignored).
- [ ] **Step 5: Commit** `bench: front-tier manifest, held-out split, scoring`.

### Task 3: Competitor adapters

**Files:**
- Create: `solver/benchmarks/adapters/cudss_adapter.cpp`, `tools/bench/wsl_ssids.md` (build recipe), `tools/bench/run_wsl.py`
- Modify: `solver/benchmarks/CMakeLists.txt` (`SMF_BENCH_CUDSS` option, default OFF), `tools/bench/run.py` (a `--runner wsl` path for SSIDS and the Linux builds)

**Interfaces:**
- Produces: solver ids `cudss`, `ssids`, `ma57` and `ma97` in records, with the same record schema as the existing adapters.
- cuDSS runs LDLᵀ in FP64, with refinement steps set to the harness's 3.
- SSIDS runs CPU-only in WSL2 Ubuntu (gfortran, the SPRAL release tarball) through a small C driver that reads the same `.mtx` and value-sequence files as `bench_replay`.

- [ ] **Step 1: USER ACTION (blocking for MA57/MA97 only).** The user requests the CoinHSL academic licence (free; licences.stfc.ac.uk). Until it arrives, the gates use SSIDS, PARDISO and MUMPS, and MA57/MA97 are added later by setting `SMF_HSL_LIBRARY`.
- [ ] **Step 2: Failing test** `BenchmarkAdapters.CudssSolvesPoisson` in `test_benchmark_adapters.cpp`, guarded by `SMF_BENCH_CUDSS`. A 2D Poisson 30×30 is solved by the cuDSS adapter with backward error ≤ 1e-12.
- [ ] **Step 3: Implement** the cuDSS adapter (CUDA toolkit plus the cuDSS Windows package; ask the user before installing either). Then write the SSIDS WSL recipe and `run_wsl.py`, whose smoke check is `ssids` solving `benchmarks/manifests/smoke.json` accurately.
- [ ] **Step 4: Run** `CTEST BenchmarkAdapters` and `python tools/bench/run_wsl.py --manifest benchmarks/manifests/smoke.json`. Expected: PASS and accurate records.
- [ ] **Step 5: Baseline.** Ask the user first (about 2–3 h single-threaded, so run it overnight). Run the whole `fronttier.json`, both tracks, all available solvers, then `score.py`. Write `docs/performance/2026-10-27-fronttier-baseline.md` with every gate as it stands and the analyse breakdown (Task 1) for the A1 cases.
- [ ] **Step 6: Commit** `bench: cuDSS and SSIDS adapters, front-tier baseline`.

---

## Part A: Analysis

### Task 4: Flat CSR tree storage

Hundreds of small vector allocations per analyse decide spd_barrier_800: 80 µs for smf against MA27's 27 µs (`2026-10-25-beating-ma27.md`).

**Files:**
- Modify: `solver/include/smf/supernode.hpp`, `assembly_tree.hpp` and `analysis.hpp`
- Modify: all users of `.children`, `.row_indices` and `child_parent_rows` (about 70 sites; largest: `factor_posdef.cpp`, `multifrontal_serial.cpp`, `parallel/task_tree.cpp`, `supernode_detection.cpp`, `solve_sparse_fwd.cpp`, `factor_indef.cpp`, `assembly_tree.cpp`)
- Tests: `test_supernode.cpp`, `test_assembly_tree.cpp`

**Interfaces:**
- Produces, on `AnalysisKeep`:
  - `std::span<const Int> children(Int s) const`
  - `std::span<const Int> front_rows(Int s) const`
  - `std::span<const Int> child_parent_rows(Int s, Int k) const`
  - backed by `child_ptr/child_idx`, `front_row_ptr/front_row_idx` and `cpr_ptr/cpr_idx`
- `Supernode` drops `children`, and `FrontalInfo` is removed. Children stay in descending postorder and front rows stay sorted.

- [ ] **Step 1: Failing test** `AssemblyTree.FlatStorageMatchesNested`. On `kkt_lqr_h20` and a 3D Poisson, the spans match a test-local copy of the old nested builder element by element.
- [ ] **Step 2: Failing test** `AssemblyTree.EdgeCases`. An n = 1 matrix, a diagonal-only n = 5 matrix (5 roots) and a matrix with an empty column all factor and solve within the gate.
- [ ] **Step 3: CTEST AssemblyTree.** Expected: compile failure.
- [ ] **Step 4: Implement.** Build the arrays with two passes (count, fill), then migrate the sites file by file.
- [ ] **Step 5: BUILD, then the full `ctest -C Release -j 2`.** Expected: all pass, including `LeanEngine` zero-allocation refactor.
- [ ] **Step 6: Measure** `bench_profile.exe --case spd_barrier_800` and `--case kkt_network_flow` before and after, and put the numbers in the commit message. Then commit `perf(solver): flat CSR storage for the supernode tree`.

### Task 5: Small-n analyse constant factors

The research found no better serial ordering algorithm, so the levers are overheads (`ordering_analysis.md`). Task 1's breakdown decides the order, largest step first.

1. **One analyse arena.** One reusable buffer for all temporaries in `symbolic_analysis.cpp`, no zero-filling of buffers that are fully overwritten, and `AnalysisKeep` arrays moved out of it, not copied.
2. **AMD input built in one pass** from the cleaned lower CSC, with AMD's elbow room of 1.2·nnz(A+Aᵀ) + 9n. The same pass's row form feeds the etree and the column counts.
3. **Assembly tree straight from AMD**, as MA27 and MA57 do: `amd_2` leaves the assembly tree in `Pe` and supervariable sizes in `Nv`. This removes the etree pass and the column counts for supervariable members.
4. **`Control::check_input`** (default true). False skips the duplicate/out-of-range scan for callers that guarantee clean input. The IPOPT plugin passes false after the first analyse of a pattern.

**Files:** `solver/src/symbolic_analysis.cpp`, `ordering_amd.cpp`, `etree.cpp`, `solver/include/smf/control.hpp`, `smf_ma97_plugin.cpp`. Test: `test_symbolic.cpp`.

- [ ] **Step 1: Failing test** `Symbolic.FastPathMatchesReference`. Each new path gives the same `perm`, `etree.parent`, column counts and supernode partition as the current path, on the six A1 optim20 matrices and a random SPD with seed 7.
- [ ] **Step 2: Failing test** `Symbolic.CheckInputFalseOnCleanInput`. Same `AnalysisKeep` as with `check_input = true`.
- [ ] **Step 3: CTEST Symbolic.** Expected: FAIL.
- [ ] **Step 4: Implement** items 1–4 in the order of Task 1's breakdown. BUILD and run the full ctest after each item, and commit each one separately (`perf(solver): <item> in analyse`).
- [ ] **Step 5: Measure** the six A1 cases (`--trials 5`, 1 thread, about 3 min). Report G-analyse (small) in the last commit message. If it still fails, report the new breakdown and stop: do not invent further levers without data.

### Task 6: Parallel nested dissection with AMD leaves and compression

**Files:**
- Create: `solver/src/ordering_parallel_nd.cpp`, `solver/include/smf/ordering_parallel_nd.hpp`, `solver/tests/test_parallel_nd.cpp`
- Modify: `symbolic_analysis.cpp` (the METIS branch, used when `num_threads > 1`), `ordering_metis.cpp` (explicit options), the CMake files

**Interfaces:**
- Produces: `bool parallel_nd_order(Int n, const std::vector<Int>& xadj, const std::vector<Int>& adjncy, const std::vector<Int>* vwgt, int num_threads, std::vector<Int>& perm);` with `perm[new] = old`.
- Method:
  - Compress vertices with identical closed adjacency, giving supervariables with weights. This matters for the 3–6 dof-per-node FEM matrices bmwcra_1 and pwtk.
  - `METIS_ComputeVertexSeparator` on the compressed graph, recursing to `ceil(log2(num_threads)) + 1` levels.
  - Each leaf is ordered on its own `std::thread`: with AMD when it has fewer than 20,000 vertices, otherwise with `METIS_NodeND`.
  - Separators are numbered last at every level. Then expand to the full graph.
  - `METIS_OPTION_SEED = 0` everywhere.

- [ ] **Step 1: Failing tests:**
  - `ParallelNd.ValidPermutation`: a 3D Poisson 30³, plus bmwcra_1 if the fixture is present, gives a permutation whose predicted flops are ≤ 1.15 × serial METIS's.
  - `ParallelNd.DisconnectedAndTiny`: two disjoint grids give an empty separator; n ≤ 2 gives the identity.
  - `ParallelNd.CompressionExpands`: a 3-dof copy of a grid compresses 3:1 and its expanded perm keeps each node's dofs adjacent.
  - `ParallelNd.DeterministicAcrossThreads`: identical perm for 1, 2 and 8 threads over 20 repetitions.
  - `ParallelNd.ConcurrentMetisIsSafe`: 8 threads × 50 `METIS_NodeND` calls equal the serial results. **If this fails, stop and report.** The fallback is process-level separation, and that decision belongs to the user.
- [ ] **Step 2: CTEST ParallelNd.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then CTEST ParallelNd and Symbolic.** Expected: PASS.
- [ ] **Step 5: Measure** bmwcra_1 and pwtk at 8 threads (ask the user first; about 5 min). Report G-analyse (large), then commit `perf(solver): parallel nested dissection with AMD leaves`.

---

## Part B: Robust and fast indefinite factorization

### Task 7: MA97-ordering-8 parity, matching plus METIS on the compressed graph

**Files:** `solver/src/ordering_matching.cpp`, `solver/include/smf/ordering.hpp`, `symbolic_analysis.cpp` (the auto KKT policy, about lines 230–300). Test: `test_ordering_metis.cpp`.

**Interfaces:**
- Produces: `enum class CompressedOrder { Amd, Metis, Auto };` and a new parameter `CompressedOrder how = CompressedOrder::Auto` on `matching_order_lower_csc`. METIS gets vertex weights equal to the pair size (2 for a 2x2 pair).
- `Auto` computes both orderings and keeps the one with fewer predicted flops, with METIS tried only when the existing `kMetisFlopsPerRow` rule allows it.
- Per Hogg–Scott–Thorne, the auto KKT policy keeps "plain AMD first, matching only when needed". Their result was that matching orderings add fill when no pivoting was needed.
- **The trigger changes from predicted to realized cost.** Today the matching ordering is adopted only when its predicted flops are ≤ 1.5× AMD's (`symbolic_analysis.cpp`). Hogg–Scott found the matching ordering routinely predicts 50–100% more flops, while AMD's prediction underestimates the real cost by up to 10× once delays inflate the fronts. So the gate rejects matching exactly where it is needed.
- New rule: keep AMD, and when a factorization's realized `delayed_pivots > 0.05·n` (IPOPT's MA97 `od_hd_reuse` threshold), re-analyse once with the matching ordering and keep that ordering for later refactors of this pattern.
- The new rule is `Control::matching_trigger = Predicted | RealizedDelays`. Both are measured in Step 5, and the default is the one that wins on tuning.

- [ ] **Step 1: Failing tests:**
  - `MetisOrdering.MatchingKeepsPairsAdjacent`: a generated KKT with 200 zero-diagonal rows; every pair is adjacent in perm.
  - `MetisOrdering.MatchingFallsBackWhenSingular`: returns false, as the AMD variant does.
  - `MetisOrdering.MatchingCutsFlopsOnNcvxqp5`: skip if the fixture is missing; METIS's predicted flops < 0.5 × the AMD-compressed ordering's.
  - `DelayedPivots.RealizedDelayTriggerReanalysesOnce`: with `matching_trigger = RealizedDelays`, a KKT whose AMD factorization delays more than 5% of n gives:
    - the first `factor()` re-analyses once (`Info::ordering_used` becomes Matching);
    - a second `factor()` with new values does not re-analyse;
    - the solve is accurate.
- [ ] **Step 2: CTEST MetisOrdering.** Expected: FAIL.
- [ ] **Step 3: Implement**, building the compressed graph once and sharing it between both orderings.
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** the hard_indef tuning cases on both tracks (about 20 min). Record delays, static pivots and time vs baseline in the commit message, then commit `perf(solver): matching + METIS compressed ordering`.

### Task 8: APTP dense front kernel

This is the core change: SSIDS-style a posteriori threshold pivoting. The algorithm follows `pivoting_robustness.md` §Q2.

- The fully-summed columns are tiled in blocks of `nb`.
- For each block column j:
  1. Back up the diagonal tile and the tiles below it.
  2. Factor the diagonal tile with **complete pivoting** in inner blocks of `ib`:
     - take the largest remaining |a|;
     - if it is diagonal, use it as a 1x1;
     - otherwise test it as the off-diagonal of a 2x2, accepted if |Δ| ≥ ½·|a_mt|²;
     - otherwise fall back to a 1x1 on the larger diagonal.
  3. Compute the L tiles below with a TRSM. Find the first column whose |l| > 1/u, and set `nelim_j = min(nelim_j, that column)`. A 2x2 pair never splits.
  4. Restore the columns from `nelim_j` on from the backups. Failed columns stay **in place**, so there is no data movement between tiles.
  5. Apply the Schur update with only the first `nelim_j` pivots.
- Failed columns go to Task 9's fallback.

**Files:**
- Create: `solver/src/ldlt_aptp.cpp`, `solver/include/smf/ldlt_aptp.hpp`, `solver/tests/test_ldlt_aptp.cpp`
- Modify: the CMake files

**Interfaces:**
- Produces:
  - `struct AptpParams { double u = 0.01; int nb = 128; int ib = 32; double small = 1e-20; };`
  - `struct AptpResult { int num_eliminated; int num_failed; InertiaCounts inertia; double flops; };`
  - `AptpResult ldlt_aptp_front(double* F, int f, int num_elig, Int* rows, const AptpParams&, int8_t* tags, double* work);` with the **same output layout and tags convention as `ldlt_factor_front`** (see `ldlt_kernel.hpp`), so Task 9 can swap it in. Failed columns come back permuted to `[num_eliminated, num_elig)`.
  - `std::size_t aptp_work_size(int f, int num_elig, const AptpParams&);` covering the backups of one block column.

- [ ] **Step 1: Failing test** `LdltAptp.MatchesTppOnRandomIndefinite`. On 200 random dense symmetric indefinite fronts (f ∈ [8, 600], num_elig ∈ [1, f], seed 11):
  - `‖P A Pᵀ − L D Lᵀ‖ / ‖A‖ ≤ 1e-13` over the eliminated block;
  - every |l| ≤ 1/u;
  - inertia equal to the eigenvalue sign count (LAPACK `dsyev`) whenever num_failed == 0.
- [ ] **Step 2: Failing test** `LdltAptp.AllColumnsFailOnZeroBlock`. A front whose eligible block is zero and has nonzero off-diagonals gives `num_eliminated == 0` and `num_failed == num_elig`, with F unchanged in content (up to the row permutation).
- [ ] **Step 3: Failing test** `LdltAptp.SaddlePointUsesTwoByTwo`. `[[0, Bᵀ],[B, 0]]` with B 50×50 well conditioned: everything is eliminated, every pivot is a 2x2, and inertia is (50, 50, 0).
- [ ] **Step 4: Failing test** `LdltAptp.SchurMatchesReference`. The Schur complement agrees with `ldlt_factor_front`'s to 1e-12 (relative) whenever both eliminate the same pivot set.
- [ ] **Step 5: CTEST LdltAptp.** Expected: compile failure.
- [ ] **Step 6: Implement.** BLAS-3 for the TRSM and the update (MKL `dgemmtr` for the lower update, as the existing kernel does). The inner complete-pivoting block stays in cache.
- [ ] **Step 7: BUILD, then CTEST LdltAptp.** Expected: PASS.
- [ ] **Step 8: Microbenchmark.** Extend `bench_front_kernel.cpp` with an `--aptp` flag, and compare against `ldlt_factor_front` for f ∈ {128, 512, 2048} on random indefinite and KKT-like fronts. Record GFLOP/s in the commit message.
- [ ] **Step 9: Commit** `feat(solver): APTP dense front kernel`.

### Task 9: APTP in the multifrontal engine with a blocked fallback

**Files:** `solver/src/multifrontal_serial.cpp`, `solver/src/ldlt_kernel.cpp`, `solver/include/smf/control.hpp`, `info.hpp`. Test: `test_factor_indef.cpp`, `test_delayed_pivots.cpp`.

**Interfaces:**
- Consumes: `ldlt_aptp_front` (Task 8).
- Produces:
  - `enum class FrontKernel { Tpp, Aptp, Auto };` and `Control::front_kernel = FrontKernel::Auto`. Auto uses APTP for fronts with `num_elig ≥ 128` and the existing kernel below that.
  - `Info::aptp_failed_columns`.
- The fallback runs the existing blocked TPP kernel `ldlt_factor_front` on the failed trailing columns, which is blocked, unlike SSIDS's serial unblocked one. Columns that still fail are delayed to the parent as today.

- [ ] **Step 1: Failing tests:**
  - `FactorIndef.AptpEngineMatchesTppAccuracy`: on every optim20 KKT case and 5 random indefinite matrices, the Aptp and Tpp engines both pass the accuracy gate, and their inertia is equal.
  - `DelayedPivots.AptpFallbackThenDelay`: a matrix built so some columns fail APTP, pass TPP, and some fail both; check `aptp_failed_columns > 0`, `delayed_pivots > 0`, accurate solve.
  - `DelayedPivots.AptpRefactorStartsClean`: a second `factor()` with benign values gives `aptp_failed_columns == 0` and `delayed_pivots == 0`.
- [ ] **Step 2: CTEST FactorIndef and DelayedPivots.** Expected: FAIL.
- [ ] **Step 3: Implement.** The work buffers come from the existing planned workspace; refactors must stay allocation-free (`LeanEngine` test).
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** hard_indef and large_kkt tuning on both tracks, plus optim20 (≥ 1.41x constraint). Tune `nb`, `ib` and the Auto cutoff of 128 on tuning only, and record the sweep in the commit message.
- [ ] **Step 6: Commit** `perf(solver): APTP front kernel in the engine`.

### Task 10: Delay budget, static pivoting as a last resort, honest inertia

**Files:** `control.hpp`, `info.hpp`, `factor_indef.cpp`, `multifrontal_serial.cpp`, `smf_ma97_plugin.cpp`, `ipm_session.cpp`. Test: `test_delayed_pivots.cpp`, `test_ma97_plugin_abi.cpp`.

**Interfaces:**
- Produces:
  - `Control::delay_budget = -1` (−1 means unlimited, which is today's behaviour). When it is ≥ 0, after that many delayed pivots the remaining fronts use the existing static path with `static_pivot_tol = 1e-8`, the MA57 approach. The 1e-8 value follows `refinement_mixed_precision.md` (√eps scale).
  - `Info::inertia_exact`: false iff `static_pivots > 0`.
  - The plugin leaves `delay_budget = -1`, and, as a guard, treats any factor with `inertia_exact == false` as MA97's "singular/fail", so IPOPT increases δ. The general API defaults stay unchanged.

- [ ] **Step 1: Failing tests:**
  - `DelayedPivots.BudgetSwitchesToStatic`: a generated KKT that delays more than 5% at u = 0.01 with `delay_budget = 10` gives `delayed_pivots ≤ 10`, `static_pivots > 0`, `inertia_exact == false`, and an accurate solve (with Task 11 refinement if it has landed, otherwise classical).
  - `DelayedPivots.BudgetResetsOnRefactor`: well-conditioned values give `inertia_exact == true` and `static_pivots == 0`.
  - `Ma97PluginAbi.InexactInertiaReportsFailure`: forcing a static pivot through the test hook makes the plugin return MA97's failure flag, never a success with wrong inertia.
- [ ] **Step 2: CTEST DelayedPivots and Ma97PluginAbi.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(solver): delay budget, static last resort, inertia_exact`.

### Task 11: FGMRES refinement escalation

**Files:** `solver/src/refinement.cpp`, `solver/include/smf/refinement.hpp`, `solver/src/solve.cpp`. Test: `test_refinement.cpp`.

**Interfaces:**
- Produces: `enum class RefineMethod { Classical, Fgmres, Auto };` and `Control::refine_method = Auto` (default), `Control::refine_max_steps = 10`, `Control::fgmres_restart = 20`.
- Auto runs classical refinement and switches to right-preconditioned FGMRES(`fgmres_restart`) with the LDLᵀ factor as preconditioner, all in FP64, when either:
  - a step's backward error contracts by less than 0.5 (stagnation), or
  - `static_pivots > 0` and classical refinement hasn't met the gate within 3 steps.
- It stops on both gate measures. It returns `RefinementResult` with a new `method_used` field and the existing fields.

- [ ] **Step 1: Failing tests:**
  - `Refinement.FgmresRecoversStaticPivots`: a KKT factored with `static_pivot_tol = 1e-6` (deliberately large) on which classical refinement stagnates. Auto reaches the gate with `method_used == Fgmres`.
  - `Refinement.SingularDoesNotLoop`: on a singular KKT (a d_pretok-like null space; use the turon_m fixture if present, otherwise a generated pressure-null-space Stokes), refinement returns in at most `refine_max_steps` total iterations with a non-success status.
  - `Refinement.ClassicalPathUnchanged`: on a well-conditioned SPD, `method_used == Classical` and the step count is identical to today's.
- [ ] **Step 2: CTEST Refinement.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use one preconditioner application per iteration through the existing solve, with modified Gram–Schmidt orthogonalization.
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure.** On hard_indef `recommended`, record static-pivot cases solved and solve + refine time against PARDISO. Then commit `feat(solver): FGMRES refinement escalation`.

---

## Part C: Top of tree, determinism, small fronts, solve

### Task 12: CHOLMOD-rule amalgamation

**Files:** `solver/src/supernode_detection.cpp` (`amalgamate_relaxed`, constants at lines 8–13). Test: `test_supernode.cpp`.

**Interfaces:**
- Produces: the merge rule becomes the current rule **or** CHOLMOD's defaults (`ordering_analysis.md`): ns ≤ 4, or no new zeros, or (ns ≤ 16 and z < 0.8), or (ns ≤ 48 and z < 0.1), or z < 0.05. Here ns is the merged pivot count and z the merged zero share. This is guarded by `Control::amalgamation = Smf | SmfPlusCholmod`, defaulting to `SmfPlusCholmod` only if Step 5 passes, otherwise `Smf`.

- [ ] **Step 1: Failing tests:**
  - `Supernode.ThinTopChainMerges`: a 20-node chain, 10 columns each, on a 1,800-row front with < 5% zeros added per merge, gives ≤ 4 supernodes and stored entries +< 5%.
  - `Supernode.CholmodRuleRespectsZeroShare`: 10% zeros per merge with ns > 48 does not merge.
- [ ] **Step 2: CTEST Supernode.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** helm3d01, copter2 and the full optim20 (≥ 1.41x constraint). Set the default as stated above and commit `perf(solver): CHOLMOD-rule amalgamation option`.

### Task 13: 2D tile tasks at the top of the tree

Today the DAG at the top runs one threaded-BLAS call per front, so a chain of thin fronts serializes. MA87, SSIDS, PaStiX and Tacho all use 2D block tasks there instead (`kernels_parallelism_gpu.md`).

**Files:** `solver/src/multifrontal_serial.cpp` (tree-parallel top phase), and new `solver/src/parallel/tile_tasks.cpp` and `.hpp`. Test: `test_tree_parallel.cpp`.

**Interfaces:**
- Consumes: the APTP block-column structure from Task 8. The tasks are `Factor(jj)`, `ApplyN(ij)` and `Update(ik, jk)` with dependencies on block columns.
- Produces: `Control::top_tiles = true` when `num_threads > 1`. Fronts with f ≥ 512 at the top of the tree are factored as tile tasks on the existing `std::thread` pool, with serial BLAS per task. The update of a child's contribution into its parent can start as soon as the child's block column is final.

- [ ] **Step 1: Failing tests:**
  - `TreeParallel.TileTasksMatchSerial`: helm3d01 if present, otherwise a generated 3D Helmholtz 30³. 8-thread tile results match 1-thread results to backward error 1e-14, with equal inertia.
  - `TreeParallel.TileTasksNoOversubscription`: the peak number of active BLAS threads (measured with the existing BLAS-thread guard) is ≤ `num_threads`.
- [ ] **Step 2: CTEST TreeParallel.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** helm3d01, copter2, bmwcra_1 and the large_kkt models at 8 and 16 threads (ask the user first; about 30 min). Then commit `perf(solver): 2D tile tasks at the top of the tree`.

### Task 14: Deterministic parallel mode

**Files:** `solver/src/multifrontal_serial.cpp`, `solve.cpp`, `solver/src/parallel/*`, `solver/include/smf/determinism.hpp`. Test: `test_parallel_determinism.cpp`.

**Interfaces:**
- Produces: with `SMF_DETERMINISTIC=ON` or `Control::deterministic = true`:
  - children are assembled into their parent in fixed postorder, not completion order;
  - each task runs with a fixed BLAS thread count, independent of how many threads are free;
  - MKL reproducibility is turned on (`mkl_cbwr_set(MKL_CBWR_AVX2)` once at first use).
  - Default off. The measured cost goes in the commit message.

- [ ] **Step 1: Failing test** `ParallelDeterminism.BitwiseAcrossThreadCounts`. Factor + solve of kkt_pde_opt, a 3D Poisson and the Task 13 Helmholtz at 1/2/8/16 threads, 10 runs each, give bitwise-identical solutions and inertia.
- [ ] **Step 2: CTEST ParallelDeterminism.** Expected: FAIL (today parallel runs vary from run to run).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** the overhead at 8 threads on large_kkt and commit `feat(solver): deterministic parallel mode`.

### Task 15: Small-front kernels

Below the 96-row blocked cutoff, small fronts dominate the very sparse KKT systems that MA27 wins.

**Files:** `solver/src/ldlt_kernel.cpp`, `solver/src/multifrontal_serial.cpp`, `solver/benchmarks/bench_front_kernel.cpp`. Test: `test_ldlt_dense.cpp`.

**Interfaces:**
- Produces:
  - AVX2 microkernels for the rank-1 and rank-2 Schur updates of fronts with f ≤ 32, as fixed-size templates for f ∈ {4, 8, 16, 32} with padded tails;
  - a fused assemble+factor path for leaf fronts, with no contribution block copy when the parent is the next front in postorder.
- Output layout is unchanged.

- [ ] **Step 1: Failing test** `LdltDense.SmallFrontKernelsMatchGeneric`. 1,000 random indefinite fronts with f ∈ [1, 32] give a bitwise-equal pivot sequence and factors within 1e-14 of the generic unblocked kernel.
- [ ] **Step 2: CTEST LdltDense.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** optim20 and the small_kkt tuning cases. Then commit `perf(solver): small-front microkernels`.

### Task 16: Solve phase, inverted diagonal blocks and blocked multi-RHS

**Files:** `solver/src/solve_forward.cpp`, `solve_backward.cpp`, `solve_diag.cpp`, `refinement.cpp`. Test: `test_solve_end_to_end.cpp`.

**Interfaces:**
- Produces:
  - `Control::cache_inverse_diag = true`: at factor time, store `L11⁻¹` for each front's diagonal block with p ≥ 16 pivots, so the solve does a GEMV/GEMM instead of a TRSV/TRSM.
  - `Solver::solve(span<double> B, int nrhs, int ldb)`, solving several right-hand sides at once with BLAS-3 per front. FGMRES and block refinement use it where they need several solves.

- [ ] **Step 1: Failing tests:**
  - `SolveEndToEnd.InverseDiagMatchesTrsv`: solutions within 1e-13 relative on optim20 and the large_kkt replays (fixtures as present).
  - `SolveEndToEnd.MultiRhsMatchesLoop`: nrhs ∈ {1, 2, 7, 32} gives the same result as nrhs separate solves, to 1e-14.
- [ ] **Step 2: CTEST SolveEndToEnd.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, then the full ctest.** Expected: PASS.
- [ ] **Step 5: Measure** IPOPT backsolve time on the 7 models against MA27 and MA57 (if licensed). Then commit `perf(solver): cached inverse diagonal blocks and multi-RHS solve`.

---

## Part D: Small, very sparse IPOPT KKT (the MA27/MA57 home ground)

### Task 17: Close the PGLib per-iteration factorization gap

smf's factorization is 1.7–2.2x MA27's on PGLib OPF (`ipm-mode-status`). Tasseff 2019 found MA27 and MA57 best on OPF, with threads not helping MA57. The report ranks this second overall, because it is the case IPOPT users hit most.

The leading hypothesis is per-front overhead in value-only refactorization: thousands of tiny fronts, each paying setup, assembly-map and workspace costs that MA27's flat loop doesn't. Step 1 confirms or refutes this before any change. The `LeanEngine` test already guarantees zero allocations per refactor, so the target is per-front fixed cost, not malloc.

**Files:** chosen by the profile, most likely `multifrontal_serial.cpp`, `ldlt_kernel.cpp`, `factor_indef.cpp` and `smf_ma97_plugin.cpp`. Test: `test_ipopt_plugin_runtime.cpp`.

- [ ] **Step 1: Profile.** Run `bench_profile --sample factor --case <pglib capture> --lines` on the three largest PGLib tuning captures. Write the top 10 line-level hotspots and the per-front overhead (fronts with f ≤ 8) into `docs/performance/triage-pglib-notes.md`.
- [ ] **Step 2: Failing test** `IpoptPluginRuntime.PglibReplayAccurate`. The three captures replay with 0 failing steps and the same inertia as MA27 (the reference inertia is recorded in the capture). This guards the changes that follow.
- [ ] **Step 3: Remove the top hotspot** from Step 1. BUILD, run the full ctest, re-profile, and commit. Repeat while the top hotspot is ≥ 10% of factor time, at most 4 rounds.
- [ ] **Step 4: Measure** IPOPT end to end on the PGLib and COPS tuning models against MA27 (and MA57 if licensed). Iteration counts must stay within ±5%. Record G-small in the last commit message.

---

## Phase E: Validate and publish

### Task 18: Held-out validation and front-tier report

**Files:** create `docs/performance/<date>-front-tier.md`; modify `docs/roadmap.md` and the memory notes.

- [ ] **Step 1: Full run.** Ask the user first; it takes several hours, so run it overnight. Run `fronttier.json` (tuning + heldout, both tracks) at 1 thread, and the large_kkt, hard_indef and spd classes at 8 threads, with all available solvers.
- [ ] **Step 2: IPOPT end to end** on PGLib, COPS and the 7 models with smf, MA27, MA57/MA97 (if licensed) and MUMPS. Report iteration counts.
- [ ] **Step 3: Write the report.** For every gate, on both splits: pass/fail, shifted geomeans, performance profiles, solved counts, accuracy failures and the inertia-exactness record. Include an honest list of where smf still loses.
- [ ] **Step 4: Commit** `docs(perf): front-tier results`.

## Not in this plan (evidence says later or never)

- **GPU factorization.** cuDSS wins only on very large, condensed or regularized systems at 1e-6. FP32 has no evidence of meeting 1e-10 on late IPM systems. Revisit after Phase E, with the quasidefinite mode from the roadmap.
- **Mixed-precision CPU factorization.** Zounon 2022 found the 2x gain only on the largest problems; not worth it for IPM KKT systems.
- **BLR / HSS compression.** No KKT/IPM evidence.
- **ParAMD.** It only pays off from about 72K rows on 64 threads. Task 6 covers the large case.

## Dependencies and parallelism

- **Priority if time is short** (the report's ranking): Phase 0, then Task 7 (KKT pipeline parity and the realized-delay trigger), Task 17 (PGLib), Tasks 8–9 (APTP), Tasks 10–11 (escalation ladder), Tasks 13–14, then Part A.
- **Phase 0 comes first.**
- **Part A (4–6), Part B (7, 8→9, 10, 11) and Tasks 12 and 15 are independent of each other.** They touch different files, except Task 4, which touches the factor and solve code that 9 and 15 also edit, so Task 4 lands before 9 and 15.
- **Task 13 needs Task 8.**
- **Task 14 needs Task 13.**
- **Task 16 is independent.**
- **Task 17 comes after 9 and 15**, because it profiles the new kernels.
- **Task 18 is last.**
