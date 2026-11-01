# Thread Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** smf never runs slower on N threads than on 1 thread, and uses only as many threads as pay off. This holds for both the factorization and the solve, on every tree shape.

**Architecture:** This plan removes three overheads of the subtree-parallel engine and adds a cost model that picks the thread count.
- A persistent worker pool replaces three rounds of `std::thread` creation per factorization.
- A spin-then-sleep top-part scheduler with chain continuation replaces the mutex, condition variable and `notify_all` on every front.
- The cost model works on the assembly tree: critical path, work per thread, and per-front hand-off cost. It picks the thread count for the factor, and separately for the solve.
- This replaces the stop-gap `Control::parallel_min_work` from d885087.

**Tech Stack:** C++20, MSVC 2022, std::thread (not OpenMP), MKL sequential BLAS, GoogleTest, Python 3.12 bench tools.

**Spec:** the Background section below, which records the measurements this plan argues from. There is no separate spec document.

## Background (the spec)

The data comes from the front-tier board, merged main 7ee12e4 plus feat/sod-next-step, smf, `benchmarks/results/fronttier_20261030-t816-merged`. Factor and solve time are per step, in ms.

| case (track) | factor 1T → 8T → 16T | solve 1T → 8T → 16T | tree |
|---|---|---|---|
| robot_arm_8k (ipopt) | 11.2 → 32.9 → 108 | 36 → 80 → 83 | 22,667 fronts, 11,804 in the top part at 16T |
| cops_elec_200 (ipopt) | 5.0 → 7.9 → 10.2 | 7.3 → 8.0 → 10.1 | 226 fronts, work bound 1.00 |
| cops_polygon_200 (ipopt) | 3.3 → 7.1 → 7.9 | 8.4 → 14.3 → 14.4 | 20,115 fronts, bound 1.04 |
| TSOPF_FS_b300_c3 (rec) | 1355 → 1012 → 1113 | — | 16T slower than 8T |
| nd24k (SPD) | 30,101 → 16,895 → 16,925 | — | bound 1.57: flat past 8T |

Profile of robot_arm_8k at 16 threads (`bench_profile --sample factor`):
- 82% of samples are in `ZwWaitForAlertByThreadId`, `RtlSleepConditionVariableSRW` and `NtAlertThreadByThreadIdEx`.
- `ZwCreateThreadEx` shows up on every factorization.

Causes, in `solver/src/multifrontal_serial.cpp`:
1. **Thread creation.** `run_pool` creates `threads - 1` `std::thread`s three times per factorization: subtrees, top part, gather.
2. **Top-part hand-offs.**
   - Phase 2 holds one mutex and one condition variable. After every front it calls `cv.notify_all()`, which wakes every idle thread.
   - About 8 µs per front, times 11.8k fronts, is about 100 ms.
   - `partition_subtrees` moves every node heavier than total/(4T) into the top part. A chain-shaped tree, such as robot_arm's time stages, therefore puts its whole spine there.
3. **Solve top part** (`solve_forward.cpp`, `solve_backward.cpp`):
   - each thread zeroes an `ntop`-row buffer per solve, and the buffers are reduced per solve;
   - every top-front update is an atomic add.
   - Both costs scale with T × (top rows).
4. **Thread count.** Measured work bound = total front work / heaviest root-to-leaf path:
   - the bound caps the speedup of fronts that are not split into tiles: ncvxqp3 ipopt has bound 1.54 and measures 0.66×, about 1/1.54;
   - `partition_subtrees` always uses all requested threads, even when 4 would do;
   - fronts run as tile tasks (u ≥ 1e-6, indefinite, at least 512 rows) are parallel inside: helm3d01 rec has bound 1.40 and still measures 0.48×.

Out of scope, as a follow-up plan: intra-front parallelism for SPD fronts and for u < 1e-6 fronts. That would be tile Cholesky on the pool and a parallel TPP trailing update. It is what lifts the 8→16 plateau on nd24k, c-big ipopt and ncvxqp3 ipopt. This plan's cost model treats those fronts as serial, which is what they are today.

## Global Constraints

- Windows, MSVC 2022, CMake. cmake and ctest are not on PATH. Their directory is `"/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/"`.
- Build directory: the worktree's `build/` (already configured). Build with `--config Release --parallel 4` at below-normal priority (`nice -n 10`). The user games on this machine.
- Tests that use CoinHSL need `SMF_HSL_LIBRARY=C:/Users/daniel/projects/opt_solver/build-deps/coinhsl-2024.5.15/CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5/bin/libhsl.dll`, with that `bin` directory on PATH.
- Threads are `std::thread` only, never OpenMP. MSVC's OpenMP runtime next to MKL or CHOLMOD aborts.
- Deterministic mode (`Control::deterministic`) keeps bitwise-identical factors, inertia and solutions at every `num_threads`. `ParallelDeterminism` and `TreeParallel` tests must pass unchanged.
- BLAS oversubscription: `blas_threads_peak()` never exceeds `num_threads` during a factor (`TreeParallel.TileTasksNoOversubscription`).
- Delay-budget regions stay as they are: `partition_subtrees(keep, kBudgetNominalThreads, top, kTreeParallelMinWork)`.
- Benchmarks:
  - run one at a time, and check that no `bench_replay.exe` is running first;
  - run at below-normal priority;
  - announce any multi-core run longer than 10 minutes.
- Never push. Never commit CoinHSL files, the `.gitignore` change, or `*.log` files.
- Code style: 2-space indent in `src/`, 4 in `tests/`. Match the surrounding comment density. Every new `Control` or `Info` field gets a comment with the evidence for its default.

## Review Focus

1. **A stop mid-top-part.** InertiaExceeded or WorkLimitExceeded can be raised while a thread is continuing a chain. Every worker must return, no new front may start after the stop is seen, and partial counts must be summed once. Tested in Task 3: `TopScheduler.StopEndsAllWorkers`.
2. **A lost wake-up in spin-then-sleep.** A front can be queued while a worker is going to sleep, which would hang. Tested in Task 3: `TopScheduler.RandomTreesStress`, 300 random trees at 8 threads under a 120 s ctest TIMEOUT.
3. **`num_threads` changing between calls on one workspace.** The sequence 4 → 8 → 4 must recreate the pool and give correct results. Tested in Task 2: `TreeParallel.PoolFollowsThreadCount`.
4. **Two workspaces factoring at once from two user threads.** There must be no shared pool and no deadlock. Tested in Task 2: `TreeParallel.TwoWorkspacesConcurrently`.
5. **A solve at a different `num_threads` than the factor** (factor at 8, solve at 2) must be correct and must use at most 2 threads. Tested in Task 5: `TreeParallel.SolveThreadsCappedByCall`.

---

### Task 1: Measurement — phase timers, per-step stats, thread-scaling report

**Files:**
- Modify: `solver/include/smf/multifrontal_serial.hpp` (SerialFactorStats), `solver/include/smf/info.hpp`, `solver/src/multifrontal_serial.cpp`, `solver/src/factor_indef.cpp`, `solver/src/factor_posdef.cpp` (copy the stats wherever `top_part_fronts` is copied)
- Modify: `solver/benchmarks/adapters/adapter.hpp` (AdapterStats), `solver/benchmarks/adapters/hsl.cpp:339` (fill them), `solver/benchmarks/bench_replay.cpp:700` (`--print-steps` line)
- Create: `tools/bench/thread_scaling.py`, `tools/bench/test_thread_scaling.py`
- Test: `solver/tests/test_tree_parallel.cpp`

**Interfaces:**
- Produces:
  - `SerialFactorStats` and `Info` gain `double subtree_seconds, top_seconds, gather_seconds` (phases 1/2/3 of the subtree-parallel engine; 0 when serial) and `int threads_used` (threads the factorization ran on; 1 when serial).
  - `AdapterStats` gains `int threads_used, parallel_subtrees, top_part_fronts; double subtree_ms, top_ms, gather_ms`.
  - The `--print-steps` line appends ` threads=%d subtrees=%d top=%d sub_ms=%.3f top_ms=%.3f gather_ms=%.3f`.
  - `thread_scaling.py --records DIR [DIR...] --manifest M [--min-ms 1.0] --output OUT.md` reports, per case-track:
    - the median factor (`refactor_ms`, else `first_factor_ms`) and `solve_refine_ms` at each thread count found;
    - the ratio of each to the 1-thread value;
    - the best thread count;
    - the flags `FAC>1T` / `SOL>1T` (ratio > 1.05 at any T > 1, 1-thread value ≥ `--min-ms`) and `16>8` (16-thread factor > 1.05 × 8-thread factor).
  - The script ends with a count of each flag. The aggregation reuses `score.aggregate`.

- [ ] **Step 1: Write the failing C++ test** in `test_tree_parallel.cpp`:

```cpp
TEST(TreeParallel, PhaseTimesReported) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    const Run serial = factor_and_solve(A, ctrl, 1);
    const Run par = factor_and_solve(A, ctrl, 4);
    EXPECT_EQ(serial.info.threads_used, 1);
    EXPECT_EQ(serial.info.subtree_seconds, 0.0);
    EXPECT_EQ(serial.info.top_seconds, 0.0);
    ASSERT_GE(par.info.parallel_subtrees, 2);
    EXPECT_GE(par.info.threads_used, 2);
    EXPECT_GT(par.info.subtree_seconds, 0.0);
    EXPECT_GT(par.info.top_seconds, 0.0);
    EXPECT_GE(par.info.gather_seconds, 0.0);
    EXPECT_LE(par.info.subtree_seconds + par.info.top_seconds + par.info.gather_seconds,
              par.info.factor_seconds);
}
```

- [ ] **Step 2: Write the failing Python test** `tools/bench/test_thread_scaling.py`.
  - Build synthetic smf records for one case-track at threads 1, 8 and 16, with factor 10 / 30 / 50 ms and solve 5 / 4 / 4 ms.
  - Assert the report row shows factor ratios `3.00` and `5.00`, best thread count `1`, and the flags `FAC>1T` and `16>8`, but not `SOL>1T`.
  - Add a second case where 8T and 16T are faster, and assert it has no flags.
  - Assert the summary line reads `FAC>1T: 1, SOL>1T: 0, 16>8: 1`.

- [ ] **Step 3: Run both and check they fail**

Run: `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 --target test_tree_parallel` (compile error: no member `threads_used`), then `python -m pytest tools/bench/test_thread_scaling.py -q` (ModuleNotFoundError).

- [ ] **Step 4: Implement**
  - Time phases 1/2/3 with `std::chrono::steady_clock` around the three `run_pool` calls in `multifrontal_factor_serial`.
  - Set `threads_used`.
  - Copy the stats through to Info and AdapterStats, and add the print fields.
  - Write `thread_scaling.py` on top of `score.aggregate`.

- [ ] **Step 5: Run the tests and check they pass**

Run: `ctest --test-dir build -C Release -R TreeParallel --output-on-failure` → `100% tests passed`; `python -m pytest tools/bench/test_thread_scaling.py -q` → `passed`.

- [ ] **Step 6: Record the baseline.** Announce the run first: about 40 minutes, multi-core.

```bash
CASES=$(python -c "import json;m=json.load(open('benchmarks/manifests/fronttier.json'));print(','.join(sorted({c['matrix_id'] for c in m['cases'] if c['class'] in ('large_kkt','hard_indef','spd')})))")
nice -n 10 python tools/bench/run.py --manifest benchmarks/manifests/fronttier.json --output benchmarks/results/thread_sched_baseline --threads 1,2,4,8,16 --trials 3 --warmups 1 --allow-short-trials --timeout 150 --cases "$CASES" --solvers smf --build-dir <worktree>/build
python tools/bench/thread_scaling.py --records benchmarks/results/thread_sched_baseline --manifest benchmarks/manifests/fronttier.json --output benchmarks/results/thread_sched_baseline/scaling.md
```

Run it with `SMF_BENCH_PARALLEL_MIN_WORK=2e7`, so the baseline is the engine without the stop-gap. Expected: `robot_arm_8k`, `cops_elec_200` and `cops_polygon_200` flagged `FAC>1T` and `SOL>1T`, and `TSOPF_FS_b300_c3` flagged `16>8`.

- [ ] **Step 7: Commit**

```bash
git add solver/include/smf/multifrontal_serial.hpp solver/include/smf/info.hpp solver/src/multifrontal_serial.cpp solver/src/factor_indef.cpp solver/src/factor_posdef.cpp solver/benchmarks tools/bench/thread_scaling.py tools/bench/test_thread_scaling.py solver/tests/test_tree_parallel.cpp
git commit -m "feat(bench): subtree-engine phase timers and thread-scaling report"
```

### Task 2: Persistent worker pool for the factorization

**Files:**
- Modify: `solver/include/smf/workspace.hpp`, `solver/src/workspace.cpp` (NumericWorkspace pool)
- Modify: `solver/include/smf/solve_pool.hpp`, `solver/src/parallel/solve_pool.cpp` (idle BLAS accounting, thread counter)
- Modify: `solver/src/multifrontal_serial.cpp` (`run_pool`)
- Test: `solver/tests/test_tree_parallel.cpp`

**Interfaces:**
- Consumes: `SolvePool` (`run(fn)`, `size()`).
- Produces:
  - `std::shared_ptr<SolvePool> NumericWorkspace::pool` and `SolvePool &NumericWorkspace::ensure_pool(int nthreads)`. These have the same contract as `SolveWorkspace::ensure_pool`: the pool is recreated when the count changes.
  - `static long long SolvePool::threads_started()`: the process-wide count of worker threads ever created.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(TreeParallel, PoolPersistsAcrossFactorizations) {
    const CscLower A = laplacian_2d(120);
    Control ctrl;
    ctrl.matrix_type = MatrixType::RealSymmetricPositiveDefinite;
    ctrl.ordering = OrderingMethod::AMD;
    ctrl.num_threads = 4;
    ctrl.parallel_min_work = 2e7;  // small test trees (Task 4 renames: auto_threads = false)
    Solver s; Info info;
    auto ak = s.analyse(A, ctrl, info);
    NumericWorkspace ws; FactorKeep fk;
    const long long t0 = SolvePool::threads_started();
    for (int r = 0; r < 3; ++r) ASSERT_EQ(s.factor(*ak, A, ctrl, info, fk, ws), FactorStatus::Success);
    ASSERT_GE(info.parallel_subtrees, 2);
    EXPECT_EQ(SolvePool::threads_started() - t0, 3);  // one pool of 4: three workers, once
}

TEST(TreeParallel, IdlePoolHoldsNoBlasThreads) {
    // After a 4-thread factor, with its pool idle: blas_threads_active()
    // equals its value before the factor.
}

TEST(TreeParallel, PoolFollowsThreadCount) {
    // One workspace: factor at 4, 8 and then 4 threads. Each solution passes
    // validate() with eta_inf < 1e-12, and threads_started() rises by
    // 3 + 7 + 3.
}

TEST(TreeParallel, TwoWorkspacesConcurrently) {
    // Two std::threads, each with its own Solver, NumericWorkspace and
    // FactorKeep, each factoring laplacian_2d(120) at 4 threads 5 times.
    // Both finish, and every solve passes validate() with eta_inf < 1e-12.
}
```

- [ ] **Step 2: Run them and check they fail**

Run: build `test_tree_parallel`. Expected: compile error, `threads_started` and `ensure_pool` are not members.

- [ ] **Step 3: Implement**
  - `NumericWorkspace::ensure_pool`, mirroring `SolveWorkspace::ensure_pool` in `workspace.cpp`.
  - `run_pool` in `multifrontal_serial.cpp` becomes `ws.ensure_pool(threads).run(fn)` under the existing caller-side `BlasIdleScope`.
  - `SolvePool::worker`: keep the thread-lifetime `BlasSerialGuard`, but wrap the spin and wait part of each loop in a `BlasIdleScope`, so an idle worker counts 0 BLAS threads and a working one counts what its job's guards set.
  - Count started threads in the `SolvePool` constructor with a `static std::atomic<long long>`.

- [ ] **Step 4: Run all tests and check they pass**

Run: `ctest --test-dir build -C Release -j 4 --output-on-failure`. Expected: `100% tests passed`, including `TileTasksNoOversubscription`.

- [ ] **Step 5: Measure**
  - Run robot_arm_8k ipopt with `SMF_BENCH_PARALLEL_MIN_WORK=2e7` at 16 threads under `bench_replay --print-steps`.
  - Record the factor time against the baseline (108 ms) in the commit message. Expect a drop of only a few ms: the thread start-up is gone, but the hand-offs are not.

- [ ] **Step 6: Commit**

```bash
git commit -am "perf(threads): persistent worker pool for the subtree-parallel factorization"
```

### Task 3: Top-part scheduler with chain continuation

**Files:**
- Create: `solver/src/parallel/top_scheduler.hpp`, `solver/src/parallel/top_scheduler.cpp`
- Modify: `solver/src/multifrontal_serial.cpp` (phase 2 uses it), `solver/CMakeLists.txt` (source), `solver/include/smf/info.hpp`, `solver/include/smf/multifrontal_serial.hpp`
- Create: `solver/tests/test_top_scheduler.cpp`. Modify `solver/tests/CMakeLists.txt`: target `test_top_scheduler`, test name `TopScheduler`, `set_tests_properties(TopScheduler PROPERTIES TIMEOUT 120)`.

**Interfaces:**
- Consumes: `SolvePool::run`, and `TileHelpers` / `TileRunner` from `parallel/tile_tasks.hpp`.
- Produces:

```cpp
namespace smf {
// Runs a DAG of top-part fronts on a pool. A node is ready once all its
// children have completed. Ready nodes share `threads` BLAS threads: a node
// taken while k others are ready gets max(1, free / (k + 1)), and the sum of
// the shares of running nodes never exceeds `threads`.
class TopScheduler final : public TileHelpers {
public:
  // parent[i]: parent of local node i (-1: root); pending counts derive from it.
  TopScheduler(std::vector<Int> parent, int threads);
  // Worker loop for thread `tid`. run(node, blas_threads, tid) -> bool
  // (false: stop everything). Returns when every node completed or after a stop.
  template <class Run> void work(int tid, Run &&run);
  void offer(TileRunner &job, int slots) override;
  void withdraw(TileRunner &job) override;
  // Completion order (topological), and how many nodes were taken from the
  // shared ready list rather than continued by the thread that readied them.
  const std::vector<std::pair<Int, int>> &completed() const;  // (node, tid)
  int queue_pops() const;
  bool stopped() const;
};
}
```

- `SerialFactorStats` and `Info` gain `int top_part_queue_pops`.

Policy, which the implementer must follow:
- **(a) Continuation.** A thread whose node completion makes the parent ready takes that parent next itself, with its share computed under the lock. Only the other initially ready nodes go on the shared list.
- **(b) Idle waiting.** An idle worker spins for up to 50 µs (`solve_pool_relax`, the `spin_until` pattern of `solve_pool.cpp`) on atomics: ready count with free threads > 0, offers > 0, all done, stopped. Then it sleeps on a condition variable and counts itself in `sleepers`.
- **(c) Wake-ups.** Queueing a node or an offer calls `notify_one()` only when `sleepers > 0`. All done, or a stop, calls `notify_all()`.
- **(d) Tile helpers.** A worker that finds offers and no runnable node helps the newest offer (`job->add_helper(); job->help();`), exactly as phase 2 does today.

- [ ] **Step 1: Write the failing unit tests** in `test_top_scheduler.cpp`. Each runs `SolvePool pool(T); pool.run([&](int tid){ sched.work(tid, run); });`.

```cpp
TEST(TopScheduler, ChainRunsInOrderWithOnePop) {
    // 1000-node chain (parent[i] = i + 1), 8 threads: every node runs once,
    // each after its child; queue_pops() == 1; completed() lists 0..999 in order.
}
TEST(TopScheduler, BinaryTreeRespectsDependenciesAndBudget) {
    // Complete binary tree, 2^10 leaves, 8 threads; run() bumps an atomic
    // `in_use` by blas_threads and records its peak.
    // peak <= 8; every parent runs after both children; queue_pops() <= 1024.
}
TEST(TopScheduler, LoneRootGetsAllThreads) {
    // A star: 64 leaves and one root, 8 threads. The root's blas_threads == 8.
}
TEST(TopScheduler, StopEndsAllWorkers) {
    // 2000-node chain plus 500 independent leaves, 8 threads; run() returns
    // false at node 700. work() returns on every thread; stopped() is true;
    // no node runs after run() saw the stop (per-node flag set before the
    // false return; later nodes start with `stopped_seen == false`).
}
TEST(TopScheduler, RandomTreesStress) {
    // 300 random trees (std::mt19937 seed 1, 1..3000 nodes, parent[i] uniform
    // in (i, n) or -1 for the last), 8 threads, run() is a no-op: each
    // completes with every node exactly once.
}
```

- [ ] **Step 2: Run them and check they fail**

Run: build `test_top_scheduler`. Expected: compile error, no `top_scheduler.hpp`.

- [ ] **Step 3: Implement `TopScheduler`**, keeping to policies (a)–(d).

- [ ] **Step 4: Run them and check they pass**

Run: `ctest --test-dir build -C Release -R TopScheduler --output-on-failure` → `100% tests passed`.

- [ ] **Step 5: Put it in phase 2**
  - Phase 2 of `multifrontal_factor_serial` builds a `TopScheduler` over the top nodes, mapping local index ↔ supernode.
  - Its `run` does what the body of the current phase-2 loop does: the `BlasThreadGuard`, the `TileCtx{blas_threads, &sched}`, `process_front`, the negatives and work counting, `diag_marks`, the `node_budget` write, and the per-thread step index.
  - `done_order` is rebuilt from `completed()` plus those per-thread step indices.
  - Delete the old mutex, condition variable and `PoolHelpers`.
  - Set `st.top_part_queue_pops`.

- [ ] **Step 6: Run all tests and check they pass**

Run: the full `ctest -j 4`. Expected: `100% tests passed`, including the `ParallelDeterminism`, `IpmHooks` and `ScaleOnDelay` parallel cases.

- [ ] **Step 7: Measure**
  - Run robot_arm_8k ipopt and cops_polygon_200 ipopt at 16 threads with `SMF_BENCH_PARALLEL_MIN_WORK=2e7` under `--print-steps`.
  - Acceptance: robot_arm_8k factor ≤ 1.5 × its 1-thread factor, against 9.6× before, and `top_ms` ≤ 5 ms. Put the numbers in the commit message.

- [ ] **Step 8: Commit**

```bash
git add solver/src/parallel/top_scheduler.* solver/src/multifrontal_serial.cpp solver/CMakeLists.txt solver/tests/test_top_scheduler.cpp solver/tests/CMakeLists.txt solver/include/smf/info.hpp solver/include/smf/multifrontal_serial.hpp
git commit -m "perf(threads): spin-then-sleep top-part scheduler with chain continuation"
```

### Task 4: Tree partition module and the thread-count model

**Files:**
- Create: `solver/src/parallel/thread_plan.hpp`, `solver/src/parallel/thread_plan.cpp`
- Modify:
  - `solver/src/multifrontal_serial.cpp`: remove `partition_subtrees` and call the new one;
  - `solver/include/smf/control.hpp`: replace `parallel_min_work` with `auto_threads`;
  - `solver/benchmarks/bench_replay.cpp`: replace the `SMF_BENCH_PARALLEL_MIN_WORK` knob with `SMF_BENCH_AUTO_THREADS` and `SMF_BENCH_THREAD_COSTS="per_thread,per_top_front"`;
  - every test line `X.parallel_min_work = 2e7;  // small test trees` becomes `X.auto_threads = false;  // small test trees`. There are 45 such lines in 14 files, and `test_tree_parallel.cpp` also has `LightTreeStaysSerial`.
- Create: `solver/tests/test_thread_plan.cpp` (target `test_thread_plan`, test `ThreadPlan`)

**Interfaces:**
- Consumes: `AnalysisKeep` (`postorder`, `supernodes[s].parent`, `front_size(s)`, `supernodes[s].width()`).
- Produces:

```cpp
namespace smf {
struct TreeView {                       // one entry per supernode
  std::span<const Int> postorder;
  std::span<const Int> parent;          // -1: root
  std::span<const double> work;         // f^2 p
  std::span<const std::uint8_t> split;  // 1: runs as tile tasks at > 1 thread
};
TreeView tree_view(const AnalysisKeep &keep, bool tiles_on,
                   std::vector<Int> &parent, std::vector<double> &work,
                   std::vector<std::uint8_t> &split);  // fills the buffers
// The current algorithm, unchanged (largest-first; nodes heavier than
// total / (4 threads) go to the top part), on arrays.
std::vector<std::pair<std::size_t, std::size_t>>
partition_subtrees(const TreeView &t, int threads, double min_work, std::vector<std::uint8_t> &top);
struct ThreadCosts { double per_thread = 1e5; double per_top_front = 5e3; };  // work units
struct ThreadPlan { int threads = 1; double predicted_speedup = 1.0; };
ThreadPlan plan_threads(const TreeView &t, int max_threads, const ThreadCosts &c);
}
```

- The model, which `plan_threads` must implement exactly:
  - The candidates are T ∈ {1, 2, 4, 8, …} ∪ {max_threads}, up to max_threads.
  - `cost(1) = total`.
  - For T > 1: `cost(T) = max(cp(T), total / T) + T * c.per_thread + top(T) * c.per_top_front`, where:
    - `cp(T)` is the heaviest root-to-leaf sum of `work[s] / (split[s] ? T : 1)`;
    - `top(T)` is the number of top nodes that `partition_subtrees(t, T, kTreeParallelMinWork, top)` marks. It is 0, and T is not a candidate, when that returns fewer than 2 subtrees.
  - Pick the T with the lowest cost. On a tie within 2%, pick the smaller T.
  - `predicted_speedup = total / cost(T)`.
- `split[s]` is 1 exactly when `top_tiles` would tile front s at more than one thread: indefinite, `pivot_u >= kWeakPivotRatio`, `front_size >= 512`, plain threshold pivoting. Use the same predicate the engine uses for `tiles_on`, factored into one function in `thread_plan.cpp` and called from both places.
- `Control::auto_threads` (default `true`):
  - `true`: the factorization runs on `plan_threads(...).threads` threads, at most `num_threads`. Fewer than 2 means serial.
  - `false`: it partitions at `num_threads` with the `kTreeParallelMinWork` floor, which is the engine before d885087.
  - The comment states the Background evidence.
- `Info::threads_used` reports the threads actually used.

- [ ] **Step 1: Write the failing unit tests** on synthetic `TreeView`s:

```cpp
TEST(ThreadPlan, ChainIsSerial) {           // 1000-node chain, work 1e6 each, max 16
    EXPECT_EQ(plan_threads(chain, 16, {}).threads, 1);
}
TEST(ThreadPlan, BalancedTreeUsesAllThreads) {  // complete binary tree, 4096 leaves of 1e7, internal 1e5
    const auto p = plan_threads(bal, 16, {});
    EXPECT_EQ(p.threads, 16);
    EXPECT_GT(p.predicted_speedup, 8.0);
}
TEST(ThreadPlan, SerialRootCapsThreads) {   // balanced, 1024 leaves of 1e6, root 1.5e9 (60% of total)
    EXPECT_LE(plan_threads(rooty, 16, {}).threads, 2);           // split[root] = 0
    EXPECT_EQ(plan_threads(rooty_split, 16, {}).threads, 16);    // split[root] = 1
}
TEST(ThreadPlan, ManyTinyTopFrontsAreSerial) {  // robot_arm-like caterpillar: 10000-node spine of 2e3
    EXPECT_EQ(plan_threads(caterpillar, 16, {}).threads, 1);     // each spine node with one 1e3 leaf
}
TEST(ThreadPlan, PartitionCoversTree) {     // laplacian_2d(120) AnalysisKeep, T = 4
    // Subtree ranges are disjoint, every node is either top or in exactly one
    // range, and every subtree's work <= total / 16 unless it is a single node.
}
```

- [ ] **Step 2: Run them and check they fail**

Run: build `test_thread_plan`. Expected: compile error, no `thread_plan.hpp`.

- [ ] **Step 3: Implement** `thread_plan.cpp`, the engine switch to `plan_threads`, `auto_threads`, `threads_used`, the bench knobs, and the test-line rename.
  - `LightTreeStaysSerial` becomes `TreeParallel.AutoThreadsPicksSerialForChain`.
  - It factors a banded SPD matrix: n = 20000, bandwidth 10, AMD.
  - Assert `threads_used == 1` and `parallel_subtrees == 0` at `num_threads = 16`. With `auto_threads = false`, assert `parallel_subtrees >= 2`.

- [ ] **Step 4: Run all tests and check they pass**

Run: the full `ctest -j 4` → `100% tests passed`, and `ThreadPlan` → passed.

- [ ] **Step 5: Commit**

```bash
git commit -am "perf(threads): pick the thread count from a tree cost model (Control::auto_threads)"
```

### Task 5: Solve thread choice

**Files:**
- Modify:
  - `solver/include/smf/factor_posdef.hpp` (FactorKeep);
  - `solver/src/multifrontal_serial.cpp`: set the field after phase 3;
  - `solver/src/solve_forward.cpp:322` and `solver/src/solve_backward.cpp:361`: use it;
  - `solver/src/parallel/thread_plan.{hpp,cpp}`.
- Test: `solver/tests/test_tree_parallel.cpp`, `solver/tests/test_thread_plan.cpp`

**Interfaces:**
- Consumes: `ThreadCosts` and the candidate rule from Task 4, and the solve layout (`solve_subtrees`, `solve_steps`, `solve_top_begin`, `solve_top_parent`, `solve_top_rows`).
- Produces:
  - `int FactorKeep::solve_threads`: 1 when the solve should run serially, otherwise the thread count it pays to use. It is set by every factorization: 1 for a serial factor.
  - `int plan_solve_threads(std::span<const double> subtree_work, std::span<const double> top_work, std::span<const Int> top_parent, std::size_t top_rows, int max_threads, const SolveCosts &c)`, with `struct SolveCosts { double per_thread = 2e4; double per_top_step = 2e3; double per_top_row = 2.0; };`.
  - Step weights are `p * f`.
  - The model:
    - `cost(1) = Σsubtree + Σtop`;
    - for T > 1: `cost(T) = max(max_subtree, Σsubtree / T) + cp_top + T * per_thread + n_top * per_top_step + T * top_rows * per_top_row`, where `cp_top` is the heaviest path through `top_parent`.
    - Candidates and tie rule as in Task 4.
- The solves use `threads = min(num_threads, fkeep.solve_threads, nsub)`. The existing layout preconditions stay.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(ThreadPlan, SolveChainTopIsSerial) {      // 2 subtrees of 1e4, top chain of 5000 steps of 50, 4000 top rows
    EXPECT_EQ(plan_solve_threads(sub, top, chain_parent, 4000, 16, {}), 1);
}
TEST(ThreadPlan, SolveBalancedSubtreesParallel) {  // 64 subtrees of 1e6, 3 top steps, 300 top rows
    EXPECT_GE(plan_solve_threads(sub64, top3, parent3, 300, 16, {}), 8);
}
TEST(TreeParallel, SolveThreadsCappedByCall) {
    // laplacian_2d(200) SPD, ND, auto_threads = false, factor at 8 threads:
    // fk.solve_threads >= 2. Solve with ctrl.num_threads = 2: eta_inf < 1e-12,
    // and SolveWorkspace::pool->size() <= 2.
}
```

- [ ] **Step 2: Run them and check they fail**

Run: build `test_thread_plan` and `test_tree_parallel`. Expected: compile errors, no `solve_threads` and no `plan_solve_threads`.

- [ ] **Step 3: Implement** `plan_solve_threads`, setting `FactorKeep::solve_threads`, and the two solve call sites. Add an env knob `SMF_BENCH_SOLVE_COSTS="per_thread,per_top_step,per_top_row"` to `bench_replay`.

- [ ] **Step 4: Run all tests and check they pass**

Run: the full `ctest -j 4` → `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git commit -am "perf(threads): choose the solve's thread count from its own cost model"
```

### Task 6: Calibrate, accept, document

**Files:**
- Create: `tools/bench/fit_thread_costs.py`, `docs/performance/2026-10-30-thread-scheduling.md`
- Modify: `solver/src/parallel/thread_plan.hpp` (the defaults of `ThreadCosts` and `SolveCosts`), `solver/include/smf/control.hpp` (the `auto_threads` comment, with the final evidence)

**Interfaces:**
- Consumes:
  - the `--print-steps` fields from Task 1 and `threads_used`;
  - `SMF_BENCH_AUTO_THREADS`, `SMF_BENCH_THREAD_COSTS` and `SMF_BENCH_SOLVE_COSTS`;
  - `thread_scaling.py`.
- Produces: `fit_thread_costs.py --truth DIR --binary EXE --manifest M --output OUT.md`.
  - For each grid point it runs every case-track's first step in `bench_replay --print-steps`, killing the process after the first step line, and reads the thread count chosen for the factor and for the solve.
  - The grid is per_thread ∈ {2e4, 5e4, 1e5, 2e5, 5e5}, per_top_front ∈ {1e3, 2e3, 5e3, 1e4, 2e4}, and for the solve per_top_step ∈ {5e2, 1e3, 2e3, 5e3} × per_top_row ∈ {0.5, 1, 2, 4}.
  - Each grid point's loss is `Σ log(measured(T_chosen) / min_T measured(T))` over case-tracks, using the 'truth' records measured with `auto_threads` off at T = 1, 2, 4, 8, 16.
  - The output lists the 5 best grid points.

- [ ] **Step 1: Record the truth data.** Announce first: about 40 minutes, multi-core. Run the Task 1 Step 6 run with `SMF_BENCH_AUTO_THREADS=0` into `benchmarks/results/thread_sched_truth`.

- [ ] **Step 2: Fit the costs**

Run: `python tools/bench/fit_thread_costs.py --truth benchmarks/results/thread_sched_truth --binary <worktree>/build/benchmarks/Release/bench_replay.exe --manifest benchmarks/manifests/fronttier.json --output benchmarks/results/thread_sched_truth/fit.md`.
- Expected: a ranked table.
- Set the defaults to the best grid point. If the best two are within 1% of each other, take the one with the larger costs.

- [ ] **Step 3: Accept.** Run the Task 1 Step 6 run with defaults (`auto_threads` on) into `benchmarks/results/thread_sched_final`, then `thread_scaling.py`. Every criterion below must hold. If one fails, return to Step 2 with that case named in the fit; do not loosen the criteria.
  - **A1:** no `FAC>1T` and no `SOL>1T` flags, with `--min-ms 1.0`.
  - **A2:** no `16>8` flags.
  - **A3:** over case-tracks with total front work ≥ 1e9, the shifted geomean (10 ms) of factor time at 8 and at 16 threads is ≤ 1.00 × the same in `thread_sched_baseline`.
  - **A4:** the full `ctest -j 4` passes.

- [ ] **Step 4: Rescore the board**
  - Combine `thread_sched_final`'s 8/16-thread smf records with the competitor records, following `benchmarks/results/fronttier_20261030-t816-merged/score_merged`.
  - Score both splits with `tools/bench/score.py`.
  - Write `docs/performance/2026-10-30-thread-scheduling.md`: the before and after tables per Background row, the fitted costs, the A1–A4 results, and the new 8/16-thread class scores.

- [ ] **Step 5: Commit**

```bash
git add tools/bench/fit_thread_costs.py solver/src/parallel/thread_plan.hpp solver/include/smf/control.hpp docs/performance/2026-10-30-thread-scheduling.md
git commit -m "perf(threads): calibrated thread-cost model; scaling write-up"
```
