# Intra-Front Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The large fronts that run serially inside today, SPD fronts and TPP fronts (pivot u < 1e-6, static-pivot, fallback and delay-budget fronts), run as tile tasks on the persistent worker pool. Then nd24k (rec), c-big (ipopt) and ncvxqp3 (ipopt) get faster from 8 to 16 threads instead of staying flat.

**Architecture:** Two new front kernels share the existing tile engine (`TileRunner`, `TileDag`, `TileHelpers` in `solver/src/parallel/tile_tasks.hpp`):
- **Tile Cholesky** (`chol_front_tiled`). A static POTRF / TRSM / SYRK-GEMM DAG with lookahead over the APTP grid (`Grid`). It covers the whole front, Schur complement included.
- **Outer-blocked TPP** (`factor_front_outer`). `ldlt_factor_front`'s pivoting runs unchanged, but its inner 16-wide panels update only the current outer block, which is `top_tile_nb` (256) columns wide. The rest of the eligible block gets one rank-≤256 update per outer block. That update, and the final Schur update, run as a flat tile DAG on the pool.

Both kernels are bitwise identical at every thread count. Front selection follows the existing APTP-tile rule:
- with more than 1 thread, at the top of the tree, for fronts of at least 512 rows;
- in deterministic mode, for every such front at every thread count.

The cost model (`thread_plan.hpp`) learns the new split fronts, with their own scaling cap.

**Tech Stack:** C++20, MSVC 2022, std::thread (no OpenMP), MKL sequential BLAS per task, GoogleTest, Python 3.12 bench tools (`tools/bench/ab_run.py`, `ab_compare.py`, `thread_scaling.py`, `score.py`, `fit_thread_costs.py`).

**Spec:** this plan's goal statement and its Background section, which follow from:
- the recommendation at `https://claude.ai/artifact/Dvh52kmg9CwrigE8smPqUL` (plan 4);
- `docs/performance/2026-10-30-thread-scheduling.md`, Follow-ups ("Intra-front parallelism for SPD fronts and u < 1e-6 fronts");
- `docs/superpowers/plans/2026-10-30-thread-scheduling.md` ("Out of scope, as a follow-up plan");
- `docs/performance/2026-11-02-factor-side-delays.md` (the current TPP kernel and its stall handling, which this plan must keep).

## Background (the spec)

Standing on 2026-11-02, per case against the best rival:
- At 8 and 16 threads smf loses on nd24k (SPD) by 1.4-1.6x to SSIDS.
- c-big ipopt and ncvxqp3 ipopt plateau from 8 to 16 threads.
- nd24k factor time is 29,819 ms at 1T, 16,740 ms at 8T and 16,768 ms at 16T (`benchmarks/results/thread_sched_final3`).

Why the plateau, from `solver/src/multifrontal_serial.cpp` and `ldlt_kernel.cpp`:
- **Who gets tiles today.** `top_tiles_on` (`thread_plan.cpp:14`) tiles only plain threshold-pivoting indefinite fronts with `pivot_u >= kWeakPivotRatio` (1e-6). SPD (`posdef`), static-pivot, in-order, fallback and budget fronts, and every front at IPOPT's u = 1e-8, go to `ldlt_factor_front` under a `BlasThreadGuard(blas_threads)` (`multifrontal_serial.cpp:1410`).
- **The rank-16 bottleneck.** `ldlt_factor_front` runs `prm.nb = indef_panel_size = 16`-wide panels, SPD included. Each panel's `deferred_update` is a rank-16 GEMM over every remaining eligible column. Only the final `schur_update` is a large-rank GEMM. On a root front where p ≈ f, almost all flops are rank-16 updates, which threaded MKL does not scale. That is the serial-inside behaviour the write-up names.
- **APTP excluded below 1e-6.** The APTP tile kernel lacks TPP's weak-1x1 rule (R42), so it is off there. The outer-blocked TPP keeps `select_pivot` itself, and with it the rule.
- **The cost model.** It counts these fronts' work as serial (`split = 0`) and tile-split work as speeding up only to `thread_split_scaling_cap` = 8 threads. So even once the fronts scale, the model must be told.

The outer-blocked TPP kernel, for the plain threshold first pass, the stall handling and the exhaustive loop of `ldlt_factor_front`:

```
os = k; oe = min(live, os + NB)                      // the outer block [os, oe)
loop as ldlt_factor_front, with pl = min(oe, ps + nb):
  inner pivots on [k, pl), W column (k - os)          // apply_pivot, unchanged
  deferred_update over columns [pl, oe) only           // rows down to f
  if stalled or k == oe:
    up.update(F, f, os, k, oe, num_elig, W)            // F[oe:f, oe:num_elig) -= L[oe:f, os:k) W[oe:num_elig, 0:k-os)^T
    then the existing stall handling (pairs / moves / exhaustive search), all columns current
    os = k; oe = min(live or num_elig, os + NB)
end: up.schur(F, f, first, k, num_elig, tags)           // the rank-k trailing update of [num_elig, f)
```

- **Why it is correct.** Swaps and the stall handling run only when no update is pending outside the outer block, so they see exactly the columns `ldlt_factor_front` sees.
- **Bitwise equal to TPP when NB covers the front.** With NB ≥ num_elig there is one outer block, `oe = num_elig`, and the kernel is `ldlt_factor_front` call for call.
- **No lookahead in v1.** A symmetric swap permutes rows of every later column, so an update running alongside the next outer block would race.

## Global Constraints

- Windows, MSVC 2022, CMake. cmake and ctest are not on PATH. Their directory is `"/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/"` (`$CMAKE`, `$CTEST` below).
- Build directory: the worktree's `build/`. Build with `--config Release --parallel 4` at below-normal priority (`nice -n 10`). The user games on this machine.
- Threads are `std::thread` only, never OpenMP. MSVC's OpenMP runtime next to MKL or CHOLMOD aborts.
- Every tile task calls BLAS single-threaded, through `TileRunner`'s `BlasSerialGuard`. `blas_threads_peak()` never exceeds `num_threads` during a factor (`TreeParallel.TileTasksNoOversubscription`).
- Deterministic mode (`Control::deterministic`) keeps bitwise-identical factors, inertia and solutions at every `num_threads`. `ParallelDeterminism` and `TreeParallel` tests pass, the existing ones unchanged except `NoTilesBelowWeakPivotU` (Task 5 says how).
- The 1-thread non-deterministic path does not change. No new kernel runs at `num_threads = 1` unless `deterministic`. A 1T board must be within noise.
- Pivoting semantics stay those of `ldlt_factor_front`: `select_pivot`, the weak-1x1 rule, and the stall pairs and moves of `dd186ad`. They are not reimplemented.
- Delay-budget regions stay `partition_subtrees(keep, kBudgetNominalThreads, top, kTreeParallelMinWork)`.
- New `Control` fields default off until their gate task flips them. Each gets a comment with the evidence for its default.
- Benchmarks:
  - A/B is interleaved from one binary on one machine (`tools/bench/ab_run.py`, variants differing only by env), scored with `ab_compare.py` and `score.py` (shift 10 ms);
  - run one at a time, at below-normal priority, after checking that no `bench_replay.exe` is running;
  - announce any multi-core run longer than 10 minutes.
- Accuracy gate: eta ≤ 1e-10, omega ≤ 1e-8. No case-track may become Inaccurate. d_pretok, turon_m and robot_arm_8k rec are Inaccurate on every solver: they do not count.
- Never push. Never commit CoinHSL files, the `.gitignore` change, or `*.log` files.
- Code style: 2-space indent in `src/` (`tile_tasks.cpp` and `ldlt_*.cpp` use 4; match the file), 4 in `tests/`. Match the surrounding comment density.

## Review Focus

1. **An SPD front that is not positive definite inside a tile:** a zero, negative, NaN or +Inf pivot in the middle of a diagonal tile. The kernel must return the first failing column, finish the DAG (later tasks become no-ops but still release), and never hang at 1, 3 or 8 threads. `process_front` then reports `NotPositiveDefinite`. Tested in Task 2: `TileCholesky.StopsAtFirstBadPivot`.
2. **A stop mid-top-part while a tiled front has helpers:** InertiaExceeded (`max_negative_pivots`) with TPP tiles at 8 threads. Every worker must return and the status must be InertiaExceeded. Tested in Task 5: `TreeParallel.TppTilesStopMidTopPart`, under the TreeParallel ctest TIMEOUT.
3. **A delay-budget retry on a tiled TPP front:** a front first factored as a plain TPP tile front, then re-assembled and static-pivoted. Rows must be restored, and the retry is tiled again. Tested in Task 5: `TreeParallel.TppTilesBudgetRetry`.
4. **Thin or square fronts:** num_elig < nb, num_elig = 1, f = num_elig (no trailing block), and f − num_elig < nb. The contract must hold in each. Tested in Task 2 (`TileCholesky.KeepsContract`) and Task 4 (`TppOuter.KeepsContractAcrossShapes`).
5. **Deterministic mode with the new kinds:** an SPD case and a u = 1e-8 indefinite case must be bitwise identical at 1, 2, 4 and 8 threads. Tested in Task 5: `ParallelDeterminism.BitwiseAcrossThreadCounts` gains the u = 1e-8 case. The SPD `poisson_3d_26` case already exists, and Task 3 asserts it tiles.

---

### Task 1: Top-front trace and the baseline

**Files:**
- Modify: `solver/src/multifrontal_serial.cpp` (phase-2 `run` lambda, around line 1410)
- Modify: `solver/src/parallel/thread_plan.hpp/.cpp` (`top_front_trace_enabled()`)
- Test: `solver/tests/test_tree_parallel.cpp`

**Interfaces:**
- Produces: `bool top_front_trace_enabled();` (thread_plan.hpp). It reads `SMF_BENCH_TOP_FRONT_TRACE=1` once, as `plan_trace_enabled` does.
- With it set, every top-part front of at least `kTopTileMinFront` rows prints one stderr line:
  `top-front s=%d f=%d elig=%d kind=%s blas=%d tiled=%d ms=%.3f`
  - kind is one of `aptp`, `chol`, `tpp`, `none`, the kernel family the front would use;
  - Task 2 onward reuse the line.

- [ ] **Step 1: Write the failing test** `TreeParallel.TopFrontTraceListsLargeFronts`.
  - Set `SMF_BENCH_TOP_FRONT_TRACE=1` with `_putenv_s` before the first call, so the read-once cache sees it.
  - Factor `laplacian_3d(30)` (a new helper next to `laplacian_2d`: 7-point stencil, diagonal 6, off-diagonals −1) as SPD at 8 threads, `auto_threads = false`.
  - Capture stderr with `testing::internal::CaptureStderr()`.
  - Assert at least one `top-front` line with `kind=none` and `f=` ≥ 512.

- [ ] **Step 2: Run it and check it fails**

Run: `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 --target test_tree_parallel && "$CTEST" --test-dir build -C Release -R TreeParallel --output-on-failure`
Expected: FAIL, no `top-front` line.

- [ ] **Step 3: Implement** the trace.
  - Time `process_front` with `PhaseClock` only when the trace is on.
  - Print under a static mutex, so lines from different threads do not interleave.

- [ ] **Step 4: Run the test and check it passes.** Same command. Expected: `100% tests passed`.

- [ ] **Step 5: Record the baseline.** About 30 minutes, multi-core: announce it first.

```bash
nice -n 10 python tools/bench/run.py --manifest benchmarks/manifests/fronttier.json \
  --output benchmarks/results/intra_front_baseline --threads 1,8,16 --trials 3 --warmups 1 \
  --allow-short-trials --timeout 600 --cases nd24k,c-big,ncvxqp3,TSOPF_FS_b300_c3,helm3d01,bratu3d,cfd2 \
  --solvers smf --build-dir build
python tools/bench/thread_scaling.py --records benchmarks/results/intra_front_baseline \
  --manifest benchmarks/manifests/fronttier.json --output benchmarks/results/intra_front_baseline/scaling.md
```

  Then run nd24k rec, c-big ipopt and ncvxqp3 ipopt at 16T once under `bench_replay --print-steps` with `SMF_BENCH_TOP_FRONT_TRACE=1 SMF_BENCH_PLAN_TRACE=1`. Write a table into the commit message with, per case:
  - the chosen T;
  - the five largest top fronts, with f, elig, kind and ms;
  - their share of `top_ms`.

  Expected: nd24k 16T ≈ 16.7 s, and its largest fronts all `kind=none`. If the largest fronts of c-big or ncvxqp3 ipopt are under 20% of their factor time, stop and report: Tasks 4-5 would not move those cases.

- [ ] **Step 6: Commit**

```bash
git add solver/src/multifrontal_serial.cpp solver/src/parallel/thread_plan.hpp solver/src/parallel/thread_plan.cpp solver/tests/test_tree_parallel.cpp
git commit -m "feat(bench): per-front trace of large top fronts (SMF_BENCH_TOP_FRONT_TRACE)"
```

### Task 2: Tile Cholesky kernel

**Files:**
- Create: `solver/src/parallel/tile_cholesky.cpp`
- Modify: `solver/src/parallel/tile_tasks.hpp` (declaration), `solver/CMakeLists.txt` (source next to `src/parallel/top_scheduler.cpp`)
- Create: `solver/tests/test_tile_cholesky.cpp`. Modify `solver/tests/CMakeLists.txt`: target `test_tile_cholesky` (include `../src` like `test_top_scheduler`), test name `TileCholesky`, `TIMEOUT 120`.

**Interfaces:**
- Consumes: `TileRunner`, `TileDag`, `TileHelpers`, `TileWorkspace` (`tile_tasks.hpp`), and `smf_dpotrf_lower`, `smf_dtrsm_right_lower_transpose`, `smf_dsyrk_lower`, `smf_dgemm_nt` (`smf/dense_kernel.hpp`).
- Produces:

```cpp
struct CholTileResult {
    int num_accepted = 0;  // num_elig on success, else the first failing column
    double flops = 0.0;    // ldlt_factor_front's count for the same pivots
};
// Cholesky of the leading num_elig columns of the f x f front F (column-major,
// ld f, lower triangle read and written) as tile tasks: on success columns
// [0, num_elig) hold L (diagonal sqrt(d), strict upper untouched) and the lower
// triangle of F[num_elig:f, num_elig:f] the Schur complement. A pivot that is
// not > 0 and finite stops it (F then unspecified). Grid as the APTP tiles:
// K blocks of nb over the eligible columns, then blocks of nb rows. Bitwise
// identical for every nthreads >= 1. Task-id spaces past kMaxTileIds: the
// same result from a serial loop over the same tile calls.
CholTileResult chol_front_tiled(double *F, int f, int num_elig, int nb, TileWorkspace &tws,
                                int nthreads, TileHelpers *helpers);
```

  The DAG: `Factor(k)` (dpotrf on the diagonal tile, then a scan of its diagonal for non-finite values); `Solve(i, k)`, i > k (trsm); `Update(i, j, k)`, k < j ≤ i < NR, over all column blocks including the non-eligible ones (syrk when i == j, gemm_nt otherwise).
  - **Dependencies:** Solve(i, k) follows Factor(k) and Update(i, k, k−1). Update(i, j, k) follows Solve(i, k), Solve(j, k) and Update(i, j, k−1). Factor(k+1) follows Update(k+1, k+1, k).
  - **Urgent tasks:** Factor, Solve, and Update(·, k+1, k).
  - **Failure:** a failed Factor sets a stop flag. Every later `execute` returns at once, and `release` runs as usual.
  - **Flops:** the sum over pivots c < num_accepted of rem + rem(rem+1), with rem = f − c − 1. This is the TPP kernel's 1x1 count.

- [ ] **Step 1: Write the failing tests** in `test_tile_cholesky.cpp`. Each runs at `nthreads ∈ {1, 3, 8}` with `helpers = nullptr` (own threads).

```cpp
TEST(TileCholesky, KeepsContract) {
    // 30 random SPD fronts A = B B^T + f I, f in [40, 700], shapes cycling
    // num_elig = f, f/2, 1, nb - 1, f - nb/2; nb in {16, 64, 256}.
    // ||A - [L;L2][L;L2]^T - [0 0; 0 S]|| / ||A|| <= 1e-13 (lower parts);
    // num_accepted == num_elig; flops == ldlt_factor_front(posdef) flops (rel 1e-12).
}
TEST(TileCholesky, MatchesTppPosdef) {
    // Same fronts: L == ldlt_factor_front(posdef) L D^{1/2} and the Schur
    // complements agree to 1e-12 relative (the TPP posdef result, rescaled).
}
TEST(TileCholesky, BitwiseAcrossThreads) {
    // f = 600, num_elig = 520, nb = 64: memcmp of F at 1, 3 and 8 threads == 0.
}
TEST(TileCholesky, StopsAtFirstBadPivot) {
    // f = 300, nb = 64, num_elig = 300: make column 150 (inside tile 2)
    // fail with a(150,150) set to -1, 0, NaN and +Inf in turn (after the
    // B B^T + f I build, the value that reaches the pivot is non-positive or
    // non-finite). num_accepted == 150 at 1, 3 and 8 threads; returns
    // (no hang under the 120 s TIMEOUT).
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 --target test_tile_cholesky`
Expected: compile error, `chol_front_tiled` undeclared.

- [ ] **Step 3: Implement `chol_front_tiled`** in `tile_cholesky.cpp`.
  - Follow `AptpTiles` in `tile_tasks.cpp`: `Grid`, a `grow_ints`-carved `pending` / ready arrays, and `TileRunner::run`.
  - `kMaxTileIds` and `grow_ints` are file-local in `tile_tasks.cpp`. Move them to an internal header `solver/src/parallel/tile_grid.hpp` together with `Grid`, and include it from both files.

- [ ] **Step 4: Run the tests and check they pass**

Run: `"$CTEST" --test-dir build -C Release -R "TileCholesky|LdltAptp" --output-on-failure`
Expected: `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add solver/src/parallel/tile_cholesky.cpp solver/src/parallel/tile_grid.hpp solver/src/parallel/tile_tasks.hpp solver/src/parallel/tile_tasks.cpp solver/CMakeLists.txt solver/tests/test_tile_cholesky.cpp solver/tests/CMakeLists.txt
git commit -m "feat(factor): tile Cholesky front kernel on the tile runner"
```

### Task 3: SPD tiles in the factorization and the cost model

**Files:**
- Modify: `solver/include/smf/control.hpp` (new fields), `solver/include/smf/info.hpp`, `solver/include/smf/multifrontal_serial.hpp` (`SerialFactorStats`)
- Modify: `solver/src/parallel/thread_plan.hpp/.cpp` (`TileKinds`, `tree_view`, `ThreadCosts`, `FactorPlanTerms`, trace)
- Modify: `solver/src/multifrontal_serial.cpp` (`process_front`, `FactorContext`, plan cache key at ~1068, `add_stats`), `solver/src/factor_posdef.cpp` (copy the stats to Info), `solver/src/factor_indef.cpp` (the same stats)
- Modify: `solver/benchmarks/bench_replay.cpp` (env knobs next to `SMF_BENCH_AUTO_THREADS`), `tools/bench/fit_thread_costs.py`, `tools/bench/test_fit_thread_costs.py`
- Test: `solver/tests/test_tree_parallel.cpp`, `solver/tests/test_thread_plan.cpp`, `solver/tests/test_parallel_determinism.cpp`

**Interfaces:**
- Consumes: `chol_front_tiled` (Task 2).
- Produces:
  - `Control`:
    - `bool top_tiles_spd = false;` SPD fronts of ≥ 512 rows at the top of the tree run as tile Cholesky. Same thread and deterministic rule as `top_tiles`.
    - `bool top_tiles_tpp = false;` used in Task 5.
    - `int thread_split_scaling_cap_dense = 8;` the scaling cap of the Cholesky and TPP split fronts.
  - `SerialFactorStats` and `Info` gain `int top_tile_chol_fronts, top_tile_tpp_fronts`. `top_tile_fronts` keeps counting every tiled front.
  - `thread_plan.hpp` replaces `bool top_tiles_on(...)` with:

```cpp
enum class TileKind : std::uint8_t { None = 0, Aptp = 1, Cholesky = 2, Tpp = 3 };
struct TileKinds {
  bool aptp = false, chol = false, tpp = false;
  bool any() const { return aptp || chol || tpp; }
  bool operator==(const TileKinds &) const = default;
};
// aptp: today's top_tiles_on. chol: ctrl.top_tiles_spd && posdef. tpp:
// ctrl.top_tiles_tpp && !posdef. All three need (num_threads > 1 || det).
TileKinds top_tile_kinds(const Control &ctrl, bool posdef, double static_pivot,
                         double min_abs_pivot, bool det);
// The kind tree_view gives a front of the factorization (plain prm, not a
// fallback / budget front): Cholesky if chol, else Aptp if aptp, else Tpp if tpp.
TileKind factorization_tile_kind(const TileKinds &k);
```

  - `TreeView::split` holds a `TileKind` value per node. `tree_view(keep, TileKinds, ...)` sets it to `factorization_tile_kind(k)` for fronts with `f >= kTopTileMinFront`, else `None`.
  - `ThreadCosts` gains `int split_cap_dense = 8`.
  - `FactorPlanTerms` gains `double split_dense_work`: the part of `split_top_work` from Cholesky / Tpp fronts.
  - `factor_plan_cost` uses `top(T) = max(cp_top, (top_total − split_work)/T + (split_work − split_dense)/min(T, split_cap) + split_dense/min(T, split_cap_dense))`.
  - `factor_plan_trace` appends ` split_dense_work=%.6g` to each `plan-factor T=` line. `fit_thread_costs.py` accepts it, and treats it as 0 when it is missing.
  - Bench env knobs: `SMF_BENCH_TOP_TILES_SPD` = 0 | 1, `SMF_BENCH_TOP_TILES_TPP` = 0 | 1, `SMF_BENCH_SPLIT_CAP_DENSE` (int).

- [ ] **Step 1: Write the failing tests**

```cpp
// test_tree_parallel.cpp
TEST(TreeParallel, SpdTilesMatchSerial) {
    const CscLower A = laplacian_3d(30);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.top_tiles_spd = true;
    const RefinedRun serial = factor_refined(A, ctrl, 1);
    if (!deterministic_mode(ctrl)) EXPECT_EQ(serial.info.top_tile_chol_fronts, 0);
    const RefinedRun par = factor_refined(A, ctrl, 8);
    ASSERT_GE(par.info.parallel_subtrees, 2);
    EXPECT_GT(par.info.top_tile_chol_fronts, 0);
    EXPECT_EQ(par.info.top_tile_chol_fronts, par.info.top_tile_fronts);
    EXPECT_EQ(par.info.num_positive, A.n);
    EXPECT_LE(par.eta, 1e-14);
    ctrl.top_tiles_spd = false;
    EXPECT_EQ(factor_refined(A, ctrl, 8).info.top_tile_fronts, 0);
}
TEST(TreeParallel, SpdTilesNotPositiveDefinite) {
    // laplacian_3d(30) with its last diagonal entry (in the root front) set to
    // -100: factor at 8 threads returns FactorStatus::NotPositiveDefinite
    // (the same status as at 1 thread), no hang.
}
// TreeParallel.TileTasksNoOversubscription: add a loop over laplacian_3d(30)
// SPD with top_tiles_spd = true at 8 and 3 threads: top_tile_chol_fronts > 0,
// peak <= threads.

// test_thread_plan.cpp
TEST(ThreadPlan, DenseSplitCapScalesPast8) {
    // One root of work 1e12 marked TileKind::Cholesky over 64 leaf subtrees of
    // 1e9: with split_cap = 8 and split_cap_dense = 16, plan_threads at
    // max_threads 16 picks 16; with split_cap_dense = 8 it picks 8
    // (the 2% tie goes to the smaller).
}
TEST(ThreadPlan, TileKindsSelection) {
    // top_tile_kinds: posdef + top_tiles_spd at 8 threads -> {chol}; at 1
    // thread -> {} unless deterministic; indefinite u = 0.01 -> {aptp};
    // u = 1e-8 + top_tiles_tpp -> {tpp}; static_pivot > 0 + top_tiles_tpp -> {tpp}.
}

// test_parallel_determinism.cpp, BitwiseAcrossThreadCounts: run the cases with
// ctrl.top_tiles_spd = true, and for poisson_3d_26 assert
// ref.info.top_tile_chol_fronts > 0 in deterministic mode (so the SPD case
// exercises the tile kernel).
```

  `test_fit_thread_costs.py`: a trace line without `split_dense_work` parses with 0, and one with it parses its value.

- [ ] **Step 2: Run them and check they fail**

Run: build `test_tree_parallel test_thread_plan test_parallel_determinism`. Expected: compile errors (`top_tiles_spd`, `TileKind` unknown).

- [ ] **Step 3: Implement**
  - **`process_front` front kernel choice.** After `prm` (plain / fallback / budget) is chosen, a front with a `TileCtx` and `f >= kTopTileMinFront` takes:
    - `Cholesky` when `cx.posdef && cx.tiles.chol`;
    - `Aptp` as today;
    - otherwise `None` (Task 5 adds Tpp).
  - `FactorContext::tile_aprm` stays. Add `TileKinds tiles` and `int tile_nb` (= `ctrl.top_tile_nb`).
  - **After a Cholesky tile front:** `kr.num_accepted = res.num_accepted`, `kr.inertia.positive = res.num_accepted`, `kr.flops = res.flops`. In the posdef post-processing, skip the `sqrt(d)` rescale for that front, but still zero the strict upper part of columns [0, p).
  - **The serial engine in deterministic mode** (`det_tc`) applies when `tiles.any()`.
  - **The plan cache key** is `TileKinds` instead of `tiles_on`.
  - **`factor_posdef.cpp`** copies `top_tile_fronts`, `top_tile_chol_fronts` and `top_tile_tpp_fronts` to Info, as `factor_indef.cpp:594` does.
  - **The new Control fields** get comments. `top_tiles_spd`: "off until Task 6's gate".

- [ ] **Step 4: Run the full suite and check it passes**

Run: `"$CTEST" --test-dir build -C Release -j 4 --output-on-failure` and `python -m pytest tools/bench -q`
Expected: `100% tests passed`; pytest `passed`.

- [ ] **Step 5: Commit**

```bash
git add solver/include/smf solver/src solver/benchmarks/bench_replay.cpp solver/tests tools/bench/fit_thread_costs.py tools/bench/test_fit_thread_costs.py
git commit -m "feat(factor): SPD top fronts as tile Cholesky (Control::top_tiles_spd, off)"
```

### Task 4: Outer-blocked TPP kernel

**Files:**
- Create: `solver/src/ldlt_internal.hpp`
- Modify: `solver/src/ldlt_kernel.cpp` (the driver, beside `ldlt_factor_front`; it reuses the file-local `select_pivot`, `apply_pivot`, `sym_swap`, `deferred_update` and `schur_update`)
- Create: `solver/src/parallel/tpp_tiles.cpp`. Modify `solver/src/parallel/tile_tasks.hpp` (declaration) and `solver/CMakeLists.txt`.
- Modify: `solver/src/parallel/tile_tasks.cpp`. `TrailingTiles` becomes the shared `tiled_trailing_update`, and `ldlt_aptp_front_tiled` calls it.
- Create: `solver/tests/test_tpp_tiles.cpp`. Modify `solver/tests/CMakeLists.txt`: target `test_tpp_tiles` (include `../src`), test name `TppOuter`, `TIMEOUT 120`.

**Interfaces:**
- Consumes: `TileRunner` and friends, `Grid` / `grow_ints` / `kMaxTileIds` from `tile_grid.hpp` (Task 2).
- Produces:

```cpp
// ldlt_internal.hpp
namespace smf::ldlt_detail {
class OuterUpdater {
public:
    // F[oe:f, oe:ne) -= L[oe:f, os:k) W[oe:ne, 0:k-os)^T, lower trapezoid
    // (W: f x NB, the inner panels' saved pivots, column c - os for pivot c).
    virtual void update(double *F, int f, int os, int k, int oe, int ne, const double *W) = 0;
    // schur_update(F, f, p0, p, ne, tags): the rank-(p - p0) trailing update.
    virtual void schur(double *F, int f, int p0, int p, int ne, const int8_t *tags) = 0;
protected:
    ~OuterUpdater() = default;
};
// ldlt_factor_front (same contract, first = 0) with outer blocks of outer_nb
// columns (algorithm in the plan's Background). work: ldlt_outer_work_size.
LdltKernelResult factor_front_outer(double *F, int f, int num_elig, Int *rows,
                                    const LdltKernelParams &prm, int8_t *tags, double *work,
                                    int outer_nb, OuterUpdater &up);
// The updater ldlt_factor_front itself would be: deferred_update's two calls / schur_update.
OuterUpdater &serial_outer_updater();
}
// smf/ldlt_kernel.hpp
// f * (max(prm.nb, outer_nb) + 2) doubles.
std::size_t ldlt_outer_work_size(int f, const LdltKernelParams &prm, int outer_nb);

// tile_tasks.hpp
// factor_front_outer with an updater that runs each outer update as tiles
// of nb x nb over columns [oe, ne) and rows [j0, f) (dgemmt on diagonal tiles,
// dgemm_nt below, the calls deferred_update makes), and the Schur update as
// tiled_trailing_update. One tile runs on the caller without a runner, as do
// all of them at nthreads == 1: the tile calls are the same at every
// nthreads, so the result is bitwise identical for every nthreads >= 1.
LdltKernelResult ldlt_factor_front_tiled(double *F, int f, int num_elig, Int *rows,
                                         const LdltKernelParams &prm, int8_t *tags, double *work,
                                         int nb, TileWorkspace &tws, int nthreads,
                                         TileHelpers *helpers);
// The trailing update of F[ne:f, ne:f] at rank p (D in the TPP/APTP layout,
// tags as both kernels write them) as FormW + tile tasks of nb.
void tiled_trailing_update(double *F, int f, int ne, int p0, int p, const int8_t *tags, int nb,
                           TileWorkspace &tws, int nthreads, TileHelpers *helpers);
```

- [ ] **Step 1: Write the failing tests** in `test_tpp_tiles.cpp`.
  - Copy `random_symmetric`, `kkt_front` and `reconstruction_error` from `test_ldlt_aptp.cpp` into the test file (same bodies).
  - Copy `weak_front` / `weak_front_tags` from `test_ldlt_dense.cpp`.

```cpp
TEST(TppOuter, OneOuterBlockIsBitwiseTpp) {
    // 40 random fronts (random_symmetric and kkt_front, f in [100, 600],
    // num_elig in [f/2, f]), u in {0.01, 1e-8}, static_pivot in {0, 1e-8},
    // min_abs_pivot in {0, 1e-8 with static}: factor_front_outer with
    // outer_nb = num_elig and serial_outer_updater() gives the same
    // num_accepted, rows, tags and memcmp-equal F as ldlt_factor_front.
}
TEST(TppOuter, KeepsContractAcrossShapes) {
    // The same fronts, plus shapes num_elig = 1, 15, f (no trailing) and
    // f - num_elig = 7, at outer_nb in {32, 64, 256}: reconstruction error
    // <= 1e-12, max |l| <= 1/u for u = 0.01, inertia from D == res.inertia,
    // num_accepted == ldlt_factor_front's on >= 90% of the fronts, and with
    // static_pivot > 0 num_accepted == num_elig.
}
TEST(TppOuter, WeakOneByOneRuleKept) {
    // weak_front cases at u = 1e-8 embedded in a 600 x 600 front (the rest
    // diagonal 1): tags from ldlt_factor_front_tiled at 4 threads equal
    // ldlt_factor_front's (the 2x2 is chosen, no weak 1x1).
}
TEST(TppOuter, StallsHandledLikeTpp) {
    // kkt_front fronts with delayed columns first (the
    // LdltDense.StalledPanelsPullInPartners construction, f = 700): same
    // num_accepted as ldlt_factor_front, reconstruction error <= 1e-12.
}
TEST(TppOuter, BitwiseAcrossThreads) {
    // f = 900, num_elig = 800, kkt_front, u = 1e-8: ldlt_factor_front_tiled
    // at nb = 64 and nthreads 1, 3, 8: rows, tags equal and F memcmp-equal.
}
```

  Also: `LdltAptp.TiledKeepsContractAndIsDeterministic` must still pass after the `tiled_trailing_update` refactor.

- [ ] **Step 2: Run them and check they fail**

Run: build `test_tpp_tiles`. Expected: compile error (`ldlt_internal.hpp` missing).

- [ ] **Step 3: Implement**
  - `factor_front_outer`: the Background algorithm, in `ldlt_kernel.cpp`.
    - Do not change `ldlt_factor_front`'s code path.
    - Below `blocked_min_front`, or with `prm.nb <= 1`, it calls `ldlt_factor_front`.
    - Only the outer-block bounds and the flush points differ from `ldlt_factor_front`. The exhaustive loop's single-pivot step (`deferred_update(F, f, ss, k, k, num_elig, W)`) runs after a flush, as a block of its own.
  - `ldlt_factor_front_tiled` and the tiled updater in `tpp_tiles.cpp`.
  - Move `TrailingTiles` into `tiled_trailing_update` in `tile_tasks.cpp`.

- [ ] **Step 4: Run the tests and check they pass**

Run: `"$CTEST" --test-dir build -C Release -R "TppOuter|LdltAptp|LdltDense|LDLT" --output-on-failure`
Expected: `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add solver/src/ldlt_internal.hpp solver/src/ldlt_kernel.cpp solver/include/smf/ldlt_kernel.hpp solver/src/parallel/tpp_tiles.cpp solver/src/parallel/tile_tasks.hpp solver/src/parallel/tile_tasks.cpp solver/CMakeLists.txt solver/tests/test_tpp_tiles.cpp solver/tests/CMakeLists.txt
git commit -m "feat(factor): outer-blocked TPP front kernel with tiled updates"
```

### Task 5: TPP tiles in the factorization

**Files:**
- Modify: `solver/src/multifrontal_serial.cpp` (`process_front`: kernel choice, `front_work_need`, budget retry), `solver/include/smf/control.hpp` (comments on `top_tiles_tpp`, `delay_budget`, `top_tiles`)
- Test: `solver/tests/test_tree_parallel.cpp`, `solver/tests/test_parallel_determinism.cpp`

**Interfaces:**
- Consumes: `ldlt_factor_front_tiled`, `ldlt_outer_work_size` (Task 4), `TileKinds` / `TileKind::Tpp` (Task 3).
- Produces: with `cx.tiles.tpp`, a front with a `TileCtx` and `f >= kTopTileMinFront` that is not Aptp-tiled or Cholesky-tiled factors with `ldlt_factor_front_tiled(F, f, num_elig, rows, *prm, tags, work, cx.tile_nb, *P.tiles, tc->nthreads, tc->helpers)`, whatever its `prm` (plain at u < 1e-6, static, in-order, fallback or budget). `top_tile_tpp_fronts` counts it.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(TreeParallel, TppTilesBelowWeakPivotU) {
    // helmholtz_case, u = 1e-8, top_tiles_tpp = true: at 4 threads
    // top_tile_tpp_fronts > 0 and == top_tile_fronts (no APTP tiles),
    // eta <= 1e-12, inertia == the 1-thread run's.
}
TEST(TreeParallel, TppTilesStaticPivot) {
    // helmholtz_case, static_pivot_tol = 1e-8 (and again with
    // static_pivot_in_order = true), top_tiles_tpp: at 8 threads
    // top_tile_tpp_fronts > 0, Success, eta <= 1e-10.
}
TEST(TreeParallel, TppTilesBudgetRetry) {
    // helmholtz_case at u = 0.5 (many delays), delay_budget = 0,
    // top_tiles_tpp = true, 8 threads: Info::delay_budget_retries > 0,
    // top_tile_tpp_fronts > 0, Success, eta <= 1e-10.
}
TEST(TreeParallel, TppTilesStopMidTopPart) {
    // helmholtz_case, u = 1e-8, top_tiles_tpp, 8 threads, max_negative_pivots
    // = 1/2 of the 1-thread run's num_negative: InertiaExceeded, returns.
}
// NoTilesBelowWeakPivotU: run it with top_tiles_tpp = false (its assertions
// unchanged); rename nothing. TileTasksNoOversubscription: add helmholtz_case
// at u = 1e-8 with top_tiles_tpp, top_tile_tpp_fronts > 0, peak <= threads.
// ParallelDeterminism.BitwiseAcrossThreadCounts: add the case
// "helmholtz_3d_30/u=1e-8" (grid_3d(30, 0.9), pivot_u = 1e-8, top_tiles_tpp)
// and assert ref.info.top_tile_tpp_fronts > 0 in deterministic mode.
```

- [ ] **Step 2: Run them and check they fail**

Run: build and `"$CTEST" --test-dir build -C Release -R "TreeParallel|ParallelDeterminism" --output-on-failure`
Expected: FAIL, `top_tile_tpp_fronts == 0`.

- [ ] **Step 3: Implement**
  - The `Tpp` branch in `process_front`, sized by `ldlt_outer_work_size` in `front_work_need`.
  - A budget retry re-runs the kernel choice with the budget `prm`: an Aptp-tiled front retried becomes a Tpp-tiled one when `cx.tiles.tpp`.
  - Update the `Control` comments of `top_tiles` and `delay_budget`, which say "TPP at 1 thread": with `top_tiles_tpp`, the large static fronts are tiled at more than 1 thread too.

- [ ] **Step 4: Run the full suite and check it passes**

Run: `"$CTEST" --test-dir build -C Release -j 4 --output-on-failure`
Expected: `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(factor): u < 1e-6 and static top fronts as tiled TPP (Control::top_tiles_tpp, off)"
```

### Task 6: SPD gate, scaling cap, default on

**Files:**
- Modify: `solver/include/smf/control.hpp` (`top_tiles_spd = true`, `thread_split_scaling_cap_dense`, with evidence comments)
- Results: `benchmarks/results/intra_front_spd_ab/` (not committed unless the repo already commits results of this kind: check `git ls-files benchmarks/results | head`)

**Interfaces:**
- Consumes: `SMF_BENCH_TOP_TILES_SPD` and `SMF_BENCH_SPLIT_CAP_DENSE` (Task 3).

- [ ] **Step 1: Find the cap.** nd24k rec with `SMF_BENCH_AUTO_THREADS=0 SMF_BENCH_TOP_TILES_SPD=1`, `bench_replay --print-steps` at 8 and 16 threads, 3 interleaved runs each.
  - If 16T is at least 10% faster than 8T: the cap is 16.
  - Otherwise it stays 8. In that case, record the finding and run the remaining gates at 8T only; the 16T plateau stays a follow-up.

- [ ] **Step 2: Run the interleaved A/B.** About 2 h, multi-core: announce it first.

```bash
nice -n 10 python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json \
  --classes spd --rounds 3 --threads 8 --output benchmarks/results/intra_front_spd_ab/t8 \
  --binary build/benchmarks/Release/bench_replay.exe \
  --variant base=SMF_BENCH_TOP_TILES_SPD=0 \
  --variant chol=SMF_BENCH_TOP_TILES_SPD=1,SMF_BENCH_SPLIT_CAP_DENSE=<cap>
# the same with --threads 16 (output .../t16) and --threads 1 (output .../t1)
python tools/bench/ab_compare.py --a benchmarks/results/intra_front_spd_ab/t16/base \
  --b benchmarks/results/intra_front_spd_ab/t16/chol --manifest benchmarks/manifests/fronttier.json \
  --metric factor --output benchmarks/results/intra_front_spd_ab/t16/compare.md
```

- [ ] **Step 3: Check the gate.** All of these must hold:
  - **nd24k rec factor at 16T:** ≤ 0.75 × base, and ≤ 0.90 × the chol variant's own 8T.
  - **spd class, both splits, factor:** shifted geomean ≤ 0.97 at 8T and at 16T. No case-track > 1.05 × base, after an 11-round recheck of any that are flagged.
  - **1T:** geomean within 0.99-1.01. The code path does not change at 1T, so anything else is an error to find.
  - **Accuracy:** no SPD case-track becomes Inaccurate.
  - **`thread_scaling.py`** on the chol variant's records at 1/8/16T: no `FAC>1T` or `16>8` flag that the base lacks.

  If a gate fails, stop and report the table. Do not flip the default.

- [ ] **Step 4: Flip the defaults.** `top_tiles_spd = true` and `thread_split_scaling_cap_dense = <cap>`. The comments cite the nd24k numbers and the class geomeans. Run `"$CTEST" --test-dir build -C Release -j 4 --output-on-failure` → `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git commit -am "perf(factor): SPD top fronts as tile Cholesky by default"
```

### Task 7: TPP gate, default on, write-up

**Files:**
- Modify: `solver/include/smf/control.hpp` (`top_tiles_tpp = true`, with evidence)
- Create: `docs/performance/2026-11-0X-intra-front-parallel.md` (the date of the run). Modify `docs/performance/2026-10-30-thread-scheduling.md`: in Follow-ups, the intra-front line points to the new write-up.

**Interfaces:**
- Consumes: `SMF_BENCH_TOP_TILES_TPP` (Task 3); the chosen `thread_split_scaling_cap_dense` (Task 6).

- [ ] **Step 1: Run the interleaved A/B** on `--classes hard_indef,large_kkt` at `--threads 8`, `16` and `1`, 3 rounds. The variants are `base=SMF_BENCH_TOP_TILES_TPP=0` and `tpp=SMF_BENCH_TOP_TILES_TPP=1`, with the Task 6 defaults on in both. About 3 h: announce it first. Compare with `ab_compare.py --metric factor` and `--metric wall`.

- [ ] **Step 2: Check the gate.** All of these must hold:
  - **c-big ipopt and ncvxqp3 ipopt factor at 16T:** ≤ 0.85 × base each, and 16T ≤ 8T on the tpp variant.
  - **ipopt track, hard_indef and large_kkt, factor:** shifted geomean ≤ 0.98 at 8T and at 16T, and ≤ 1.00 on the recommended track. No case-track > 1.05 × base after an 11-round recheck. Expect cops_elec_200 to need a look: delay-heavy split fronts, see `per_split_front_thread`.
  - **1T:** within 0.99-1.01.
  - **Accuracy:** no new Inaccurate case-track. Report every backward error that moves more than 10x.
  - **`thread_scaling.py`:** no new `FAC>1T` or `16>8` flags.

  If the model picks a T that loses (a `FAC>1T` flag on a tpp-split case), refit with `fit_thread_costs.py`, with the split-dense terms in the traces, before flipping. If a gate still fails, stop and report.

- [ ] **Step 3: Flip the default** `top_tiles_tpp = true`, then run the full ctest → `100% tests passed`.

- [ ] **Step 4: Score the board.** Run the front-tier board at 1/8/16T for smf on the spd, hard_indef and large_kkt classes, 3 trials. Score it with `score.py --shift-ms 10` against the competitor records used in `thread_sched_board`. Then write the performance doc in the house style of `2026-10-30-thread-scheduling.md`:
  - Result table: nd24k, c-big ipopt, ncvxqp3 ipopt at 1 → 8 → 16T, before and after;
  - What changed, with commits;
  - The A/B tables;
  - The board deltas;
  - Measuring notes: one binary, interleaved, the CCD caveat;
  - Follow-ups: lookahead across outer blocks via deferred row swaps; the TPP tiled kernel for the APTP fallback columns (`first > 0`); tile Cholesky at 1T, if the 1T path should change at all (cfd2).

- [ ] **Step 5: Commit**

```bash
git add solver/include/smf/control.hpp docs/performance/
git commit -m "perf(factor): tiled TPP top fronts by default; intra-front parallelism write-up"
```
