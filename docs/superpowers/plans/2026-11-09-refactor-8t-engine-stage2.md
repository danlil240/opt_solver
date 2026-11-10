# 8-thread refactorization engine, Stage 2: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** smf's projected 8-thread run time is below MKL PARDISO's on all 8 PR #64 cases. After Stage 1 the open
cases are acopf_grid_80 (1.174), acopf_grid_160 (1.261) and obstacle_surface_200 (1.045). The Stage 2 gate (Task 10)
measures this.

**Architecture:** Stage 2 removes the serial work on the top-of-tree critical path, in three parts:
- **Part 5:** a large top front is assembled in column blocks by idle pool threads.
- **Part 7:** top fronts are taken by remaining path cost, and threads that come free go to the running front on the
  critical path. The front's next BLAS call uses its larger share.
- **Part 6:** the TPP kernel factors panel k+1 while panel k's update and Schur chunk run on the other threads.

The parts land in the spec's order 5, 7, 6. Each part is its own stacked PR on Stage 1's last PR (#69), with its own
on/off control.

**Tech stack:** as in Stage 1:
- C++20 with MSVC 2022, threaded MKL (`build-mtt`) and GoogleTest.
- Python 3 with pytest for the tools.
- The 12900K for every suite and benchmark.

**Spec:** `docs/superpowers/specs/2026-11-09-refactor-8t-engine-design.md`, sections "Stage 2", "Testing, switches and
fallbacks" and "Delivery". The Stage 1 gate report is `docs/performance/2026-11-09-8t-engine-stage1-gate.md`. Daniel
chose Stage 2 over squeezing Stage 1 on 2026-11-09.

## Global constraints

These come from the spec. Every task implicitly includes them.

- No loss at 1 or 16 threads. No loss of accuracy.
- Accurate threshold pivoting stays the default. No static pivoting by default.
- Deterministic mode keeps working: its results don't depend on the thread count.
- CPU only and MIT licence. No new third-party dependency.
- Every part does nothing at 1 thread.
- Part 5 is bit-for-bit identical at every thread count, and may run in deterministic mode.
- Part 6 runs only on large top fronts, in non-deterministic mode, at a share of 2 or more threads. Deterministic mode
  and every other front keep today's kernel.
- Part 7 keeps continuation: the thread that readies a parent runs it.
- Every part has a `Control` flag and an `SMF_BENCH_*` environment override. The overrides are read in
  `solver/benchmarks/bench_replay.cpp`, next to the existing `SMF_BENCH_*` reads (around line 680-850). A flag
  defaults to on only after its part's A/B, and then stays as the off-switch.
- Benchmarks and suites run on the 12900K only, one at a time, after checking that it is idle.
- Builds stay local, at below-normal priority, with at most 6 jobs.
- Never commit or copy the licensed CoinHSL (`build-deps/coinhsl-*`, `C:\smfkit\coinhsl_bin`) or the Mittelmann
  `.mod`/`.nl` files.
- No force-push. A fix to a lower branch is committed there and merged up the stack.
- Update `STATUS.md` (header, affected section, dated changelog line) after every PR, gate and plan change.

## Plan decisions

These are decisions the spec leaves open or that the code changes. The executor carries them, and a later finding
that contradicts one is ledgered as a ruling.

1. **Stage 2 builds on Stage 1's tip.** The branches stack on `perf/8t-p3-layout`: `perf/8t-p5-assembly`, then
   `perf/8t-p7-shares`, then `perf/8t-p6-lookahead`. Each PR targets the branch before it.
2. **Kernel times come from the phase trace, not the probe branch.**
   - The spec's kernel split (assembly 7.9 ms, panel 6.9 ms on acopf_grid_160) came from `probe/front-trace-8t`, which
     is Windows-only and not for merge.
   - Task 0 adds per-factorization sums of top-front kernel time to `SMF_BENCH_PHASE_TRACE=2`.
   - These sums are over all top fronts, not just the critical path. On acopf_grid_160 the critical path is 0.94 of
     the top span, so the sums track it.
   - The top span itself is the existing `mf_p2->mf_p3` segment.
3. **No deferred interchanges in Part 6.** The spec says panel k+1's interchanges are recorded and applied later. The
   kernel makes that unnecessary:
   - `select_pivot` takes candidates and 2x2 partners only inside the panel (`r_best >= limit` is rejected,
     `ldlt_kernel.cpp` around line 151).
   - `sym_swap(p, q)` with p, q in panel k+1 writes:
     - rows p and q of columns left of p;
     - columns p and q themselves;
     - its own W buffer.
   - The concurrent work for panel k touches none of those. It writes columns at or past the end of panel k+1, rows
     at or past it, and the Schur scratch in rows of panel k, columns >= num_elig. It reads panel k's L in rows at or
     past the end of panel k+1, and panel k's W.
   - The test that pins this is `LdltLookahead.RepeatedRunsAgree` (Task 7).
4. **The memory baseline for Stage 2 is Stage 1's tip** (`e8t_gate1`), so each part is judged on its own change.
   - Stage 1's peak-RSS rise on the three newly parallel small cases is a Stage 1 item.
   - The gate report records it against `e8t_base` too. Daniel decides it.
5. **obstacle_surface_200 must also pass the Stage 2 gate.** The spec expected it to pass after Stage 1, so it has no
   Stage 2 target of its own. Its 8T top span (about 2.6 of 6.1 ms) is the lever, so its top-span change is reported
   per part.

## Review focus

These are inputs the spec implies but no part's main test exercises. Each line names the test that pins it and the
task that owns it.

1. **A top front with a child whose row map is not increasing** (delayed rows, or the defensive extra rows). Expect
   the serial assembly order, bit for bit.
   Test: `FrontAssembly.NonMonotoneChildKeepsSerialOrder`, Task 1.
2. **A delay-budget front that assembles twice** (`delay_budget >= 0`, the retry loop in `process_front`), with
   parallel assembly on. Expect results identical to assembly off.
   Test: `TreeParallel.ParallelAssemblyWithDelayBudget`, Task 2.
3. **A factorization that stops** (`InertiaExceeded` under an inertia limit) while Parts 5-7 are on. Expect no hang,
   no stale share, and a correct next factorization in the same `FactorKeep`.
   Test: `TreeParallel.StopWithStage2PartsOn`, Task 8.
4. **A share grant racing with the front's own finish.** Expect each thread returned exactly once, and the sum of the
   running shares never above `threads`.
   Test: `TopScheduler.GrantRacingFinishKeepsBudget`, Task 4.
5. **Lookahead edge shapes:**
   - the first panel stalls;
   - num_elig is not a multiple of nb;
   - num_elig == f (no Schur rows);
   - fewer than two panels;
   - an all-zero eligible block (every column delayed).

   Expect the same pivots as today's kernel.
   Test: `LdltLookahead.EdgeShapes`, Task 7.

## Conventions used by every task

**Worktree.** `WT` = `C:\Users\daniel\projects\opt_solver-wt\engine8t-p2`. It already has a full `build-mtt`.

**Build.** `cmd //c "%USERPROFILE%\.claude\smf-remote\build_mtt.cmd" <WT> [target1+target2...]`. Join several
targets with `+`; a `;` builds only the first. Expected last line: `ALL_OK`.

**Tests.**
- Full suite: `bash ~/.claude/smf-remote/rtest.sh $WT/build-mtt "<regex>"`. Exit 0 means pass.
- A quick local run of one gtest is allowed:
  `PATH=/c/Users/daniel/mamba/envs/smfdeps/Library/bin:$PATH $WT/build-mtt/tests/Release/<exe> --gtest_filter=<name>`.
- Known flaky, pre-existing: `SolveEndToEnd` pde_control_250 at 4T (inverse vs trsv, about 1.1e-13 against a 1e-13
  tolerance). Rerun it once before counting it as a failure.

**8-case A/B** (as in Stage 1):

```bash
RRUN_SRC=$WT bash ~/.claude/smf-remote/rab.sh e8t $WT/build-mtt/benchmarks/Release/bench_replay.exe <out> - -- \
  --manifest benchmarks/manifests/probe_refactor8t.json --threads 8 --rounds 3 --seq-steps all --timeout 300 \
  --variant off=SMF_BENCH_<FLAG>=0 --variant on=SMF_BENCH_<FLAG>=1
python $WT/tools/bench/ab_compare.py --a benchmarks/results/remote/<out>/off --b benchmarks/results/remote/<out>/on \
  --manifest $WT/benchmarks/manifests/probe_refactor8t.json --metric refactor
python $WT/tools/bench/project_run.py --smf benchmarks/results/remote/<out>/on \
  --rivals benchmarks/results/remote/e8t_gate1 --manifest $WT/benchmarks/manifests/probe_refactor8t.json --threads 8
```

**Phase split with kernel sums:**

```bash
bash ~/.claude/smf-remote/rpy.sh <out> "SMF_BENCH_PHASE_TRACE=2 <flags>" tools/bench/phase_run.py <tag> -- --threads 1,8
python $WT/tools/bench/phase_split.py benchmarks/results/remote/<out> --segments mf_p2->mf_p3,sum:top_asm,sum:top_panel,sum:top_defer,sum:top_schur
```

**Results** land in `C:\Users\daniel\projects\opt_solver\benchmarks\results\remote\<out>`, which is not committed.

---

### Task 0: kernel sums in the phase trace, and the Stage 2 baseline (branch `perf/8t-p5-assembly`)

**Files:**
- Modify: `solver/src/phase_trace.hpp`, `solver/src/phase_trace.cpp`, `solver/src/multifrontal_serial.cpp` (the top
  phase's run lambda and `process_front`'s assembly), `solver/src/ldlt_kernel.cpp` (the blocked path).
- Modify: `tools/bench/phase_split.py`, `tools/bench/test_phase_split.py`.
- Create: this plan's commit.

**Interfaces:**
- Produces in `phase_trace.hpp`:

```cpp
// SMF_BENCH_PHASE_TRACE level (read once): 0 off, 1 marks, 2 marks plus kernel sums.
int phase_trace_level() noexcept;
enum class PhaseSum : int { TopAsm, TopPanel, TopDefer, TopSchur, Count };
// Adds seconds to this factorization's sum (atomic; no-op below level 2).
void phase_sum_add(PhaseSum slot, double seconds) noexcept;
// Prints "# sum <name> s=<seconds>" for every slot and zeroes them (called at mf_p3).
void phase_sum_flush() noexcept;
// RAII, level 2 only: marks this thread as running a top front, so the TPP kernel's laps are summed.
class PhaseTopFrontScope { public: PhaseTopFrontScope() noexcept; ~PhaseTopFrontScope() noexcept; };
bool phase_in_top_front() noexcept;
```

- `phase_split.split()` puts each `# sum <name> s=<v>` line into the current factorization's dict as
  `sum:<name>` in ms.
- The slot names printed: `top_asm`, `top_panel`, `top_defer`, `top_schur`.

- [ ] **Step 1: Commit this plan** on `perf/8t-p5-assembly`: `docs: 8T engine Stage 2 plan`.
- [ ] **Step 2: Write the failing tool test** in `test_phase_split.py`:

```python
def test_sum_lines_join_their_factorization():
    lines = ["# seg fi_entry t=1.000", "# seg mf_p2 t=1.001", "# sum top_asm s=0.0005",
             "# sum top_panel s=0.0002", "# seg mf_p3 t=1.004", "# seg fi_end t=1.005",
             "# seg fi_entry t=2.000", "# seg fi_end t=2.001"]
    out = phase_split.split(lines)
    assert out[0]["sum:top_asm"] == pytest.approx(0.5)
    assert out[0]["sum:top_panel"] == pytest.approx(0.2)
    assert "sum:top_asm" not in out[1]
```

- [ ] **Step 3:** Run `python -m pytest tools/bench/test_phase_split.py -q`. Expected: FAIL on the new test.
- [ ] **Step 4:** Implement the `# sum` parsing and `--segments` support for `sum:` names. Run the test again.
  Expected: PASS.
- [ ] **Step 5: Implement the library side.**
  - In `process_front`, time the assembly section (zero fill through the last child) when `tc != nullptr` and
    `phase_trace_level() >= 2`. Add it to `TopAsm`.
  - In `ldlt_factor_front`'s blocked path, when `phase_in_top_front()`, lap a steady clock:
    - select and apply go to `TopPanel`;
    - `deferred_update` goes to `TopDefer`;
    - `schur_update` goes to `TopSchur`;
    - the stall handling goes to `TopPanel`.
  - The run lambda in the top phase holds a `PhaseTopFrontScope` around `process_front`.
  - Call `phase_sum_flush()` right after `phase_mark("mf_p3")`.
  - With the trace off or at level 1, the only new cost is one cached-int compare per front and one per panel.
- [ ] **Step 6:** Build `bench_replay+test_tree_parallel`. Run `rtest.sh $WT/build-mtt "TreeParallel|PhaseTrace"`.
  Expected: exit 0.
- [ ] **Step 7: Baseline.**
  - Run the phase split at the stage tip with `SMF_BENCH_PHASE_TRACE=2`, out `e8t_s2_base_phase`.
  - Expected, acopf_grid_160 at 8T:
    - `mf_p2->mf_p3` within 20% of the gate run's top span. The spec's pre-Stage-1 figure was 22.08 ms.
    - `sum:top_asm` about 8 ms and `sum:top_panel` about 7 ms (spec: 7.9 / 6.9).
  - If the sums are off by more than 30%, find out why before Part 5. A wrong trace would mis-steer every target
    below.
  - Record in the ledger, per case at 8T: `mf_p2->mf_p3` and the four sums. The acopf_grid_160 top span is the stop
    rule's base.
- [ ] **Step 8:** Commit: `bench: top-front kernel sums in the phase trace (8T engine Stage 2 harness)`.

### Part 5: parallel assembly of large top fronts (branch `perf/8t-p5-assembly`)

### Task 1: the column-block assembly routine

**Files:**
- Create: `solver/src/front_assembly.hpp`, `solver/src/front_assembly.cpp` (add to `solver/CMakeLists.txt`).
- Test: `solver/tests/test_front_assembly.cpp` (add to `solver/tests/CMakeLists.txt`).

**Interfaces:**
- Consumes `TileRunner`, `TileDag` and `TileHelpers` from `solver/src/parallel/tile_tasks.hpp`.
- Produces:

```cpp
namespace smf {
// One child's contribution block, as process_front reads it.
struct AssemblyChild {
  const Int *map = nullptr;       // front position of each of the q rows; strictly increasing
  const double *values = nullptr; // packed lower block (ld == 0), or first column of a strided block
  std::size_t ld = 0;             // > 0: strided (small-front buffer), column jj starts at values + jj*(ld+1)
  Int q = 0;
};
// The front's own pivot columns' original entries (lower CSC): column j goes to front column pos[j],
// row ari[k] to pos[ari[k]].
struct AssemblyOriginal {
  const Int *acp = nullptr, *ari = nullptr, *pos = nullptr;
  const double *av = nullptr;
  Int col_start = 0, col_end = 0;
};
// Columns [c0, c1) of the f x f front F (leading dimension f, lower triangle): zero fill, original
// entries, then each child in order. Each entry gets the serial assembly's additions in the same order.
void assemble_front_columns(double *F, int f, const AssemblyOriginal &orig,
                            std::span<const AssemblyChild> kids, int c0, int c1);
// The whole front in 4 * nthreads column blocks of equal lower-triangle area, run as independent
// tasks by a TileRunner on nthreads threads (helpers, or the runner's own threads when null).
// nthreads <= 1: assemble_front_columns(F, f, orig, kids, 0, f) on the caller.
void assemble_front_parallel(double *F, int f, const AssemblyOriginal &orig,
                             std::span<const AssemblyChild> kids, int nthreads, TileHelpers *helpers);
}  // namespace smf
```

- Also produces `bool child_maps_increasing(std::span<const Int> map) noexcept;` (strictly increasing). The caller
  in Task 2 uses it to choose between the parallel path and today's loops.
- In a block, a child's columns are the contiguous `jj` range whose `map[jj]` lies in `[c0, c1)`. The map is
  increasing, so find the range by binary search.

- [ ] **Step 1: Write the failing tests** in `test_front_assembly.cpp`:

```cpp
// A synthetic front: f = 700, original entries in columns [0, 40), three children with q = 300,
// 450 and 120, increasing maps drawn from a fixed-seed RNG, one child strided (ld = 128 > 0).
// Values are random in [-1, 1].
TEST(FrontAssembly, ColumnBlocksEqualOneBlock) {
  // assemble_front_columns over [0, 700) vs. over the blocks {[0,1), [1,333), [333,700)}:
  // std::memcmp of the lower triangle == 0.
}
TEST(FrontAssembly, ParallelEqualsSerialAtEveryThreadCount) {
  // assemble_front_parallel at nthreads 2, 4 and 8 (helpers == nullptr) vs. assemble_front_columns
  // over [0, f): std::memcmp == 0. Also equal to a naive reference in the test (zero, originals,
  // then each child's (ii >= jj) entries in jj-outer order).
}
TEST(FrontAssembly, NonMonotoneChildKeepsSerialOrder) {
  // AssemblyChild requires an increasing map. Pin that the caller's check (Task 2's
  // child_maps_increasing) returns false for a map with one descent, and true for an increasing one.
}
```

- [ ] **Step 2:** Build `test_front_assembly`, then run it. Expected: FAIL to compile (no `front_assembly.hpp`).
- [ ] **Step 3:** Implement `front_assembly.hpp/.cpp`. The DAG is `num_blocks` root tasks with no successors, and
  `urgent()` returns false.
- [ ] **Step 4:** Run `test_front_assembly`. Expected: 3/3 pass.
- [ ] **Step 5:** Commit: `perf: column-block front assembly (8T engine Part 5)`.

### Task 2: parallel assembly in `process_front`

**Files:**
- Modify: `solver/src/multifrontal_serial.cpp` (`process_front`'s assembly section, `FactorContext`,
  `SerialFactorStats`).
- Modify: `solver/include/smf/control.hpp`, `solver/include/smf/info.hpp`, `solver/benchmarks/bench_replay.cpp`.
- Test: `solver/tests/test_tree_parallel.cpp`. Move `factor_in_step_order` from `test_parallel_determinism.cpp` into
  `solver/tests/kkt_test_matrices.hpp` (inline) so both files use it.

**Interfaces:**
- Consumes Task 1's `assemble_front_parallel`, `child_maps_increasing`, `AssemblyChild` and `AssemblyOriginal`.
- Produces:
  - `Control`: `bool parallel_front_assembly = false;` and `int parallel_assembly_min_front = 1024;`.
  - Environment: `SMF_BENCH_PARALLEL_ASSEMBLY` and `SMF_BENCH_PARALLEL_ASSEMBLY_MIN`.
  - `Info`: `int parallel_assembly_fronts = 0;`, the top fronts assembled in parallel this factorization.
  - `FactorContext`: `bool det = false;`, set from `deterministic_mode(ctrl)`. Task 8 uses it too.
- A front assembles in parallel when all of these hold:
  - `ctrl.parallel_front_assembly` is on;
  - `tc != nullptr` and `tc->nthreads >= 2`;
  - `f >= ctrl.parallel_assembly_min_front`;
  - it is not a small front;
  - every child's map is increasing.

  Otherwise today's loops run unchanged.
- The caller builds all children's maps into `P.cmap`, grown to the sum of q, with the offsets per child. It keeps
  computing `new_cv_top` / `new_cr_top` itself.
- Helpers come from `tc->helpers`. This is the scheduler's `offer`/`withdraw`, the same mechanism the tile tasks use.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST(TreeParallel, ParallelAssemblyBitwiseInDeterministicMode) {
  // grid_kkt(200, 7) and opf_like_kkt(20000, 8000, 3); deterministic mode; num_threads 2, 4, 8;
  // parallel_assembly_min_front = 64. On vs off: factor_in_step_order(fk) and the solution x
  // compare bitwise (std::memcmp); inertia equal; Info::parallel_assembly_fronts > 0 when on, 0 off.
}
TEST(TreeParallel, ParallelAssemblyWithDelayBudget) {
  // cascade_kkt(20, 150, ...) as in ReplansWhenDelaysGrowFronts, delay_budget = 0 (budget fronts
  // retry assembly), deterministic mode, 8 threads, min front 64: on vs off bitwise as above.
}
```

- [ ] **Step 2:** Run them. Expected: FAIL to compile (no `parallel_front_assembly`).
- [ ] **Step 3:** Implement the flag, the env reads, the condition, the map pre-pass, the call, and the counter.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "TreeParallel|FrontAssembly|ParallelDeterminism|RepeatedFactor"`.
  Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: large top fronts assemble in column blocks on idle threads (8T engine Part 5)`.

### Task 3: Part 5 sweep, A/B and PR

- [ ] **Step 1: Sweep `f_min`.**
  - Run the 8-case A/B with these variants:
    - `off=SMF_BENCH_PARALLEL_ASSEMBLY=0`;
    - `m512=...=1 SMF_BENCH_PARALLEL_ASSEMBLY_MIN=512`;
    - `m1024=...=1 ..._MIN=1024`;
    - `m2048=...=1 ..._MIN=2048`.
  - Pick the variant with the best geomean refactor ratio over the 8 cases.
  - Ledger the choice as a ruling, with the per-case ratios.
- [ ] **Step 2: Measure the target.** Run the phase split with the chosen min, out `e8t_p5_phase`.
  - Expected: `sum:top_asm` on acopf_grid_160 at 8T at most half of Task 0's (spec target: 7.9 -> about 2 ms).
  - Expected: `mf_p2->mf_p3` down on acopf_grid_160, acopf_grid_80 and obstacle_surface_200.
  - Expected: no case's refactor median more than 1% slower.
- [ ] **Step 3:** Make the chosen min the default and switch `parallel_front_assembly` on.
- [ ] **Step 4:** Run `rtest.sh` on `build-mtt`, and on `build-seq` (`build_mtt.cmd <WT> all sequential`).
  Expected: both exit 0.
- [ ] **Step 5:** Push and open the PR against `perf/8t-p3-layout`: "perf: parallel assembly of large top fronts (8T
  engine Part 5)". The body has the A/B table and the kernel sums. Update `STATUS.md`.

### Part 7: thread shares that follow the critical path (branch `perf/8t-p7-shares`, from `perf/8t-p5-assembly`)

### Task 4: priorities, proportional shares and growing shares in `TopScheduler`

**Files:**
- Modify: `solver/src/parallel/top_scheduler.hpp`, `solver/src/parallel/top_scheduler.cpp`.
- Test: `solver/tests/test_top_scheduler.cpp`.

**Interfaces:**
- Produces:

```cpp
// priority (empty: today's behaviour exactly, LIFO list and equal shares): one value per local node,
// its remaining path cost to the root.
TopScheduler(std::vector<Int> parent, int threads, std::vector<double> priority = {});
// The live BLAS share of a node while run(node, ...) runs; Part 7 grows it.
const std::atomic<int> &share(Int node) const;
// Sum of the running nodes' shares (under the lock). For tests.
int shares_in_use() const;
```

- With priorities:
  - The ready list is a max-heap on priority. Ties go to the higher local id, as LIFO gives today.
  - **Start share.** A node taken with free threads F and ready set R (the node included) gets
    `clamp(floor(F * p / sum_R p), 1, max(1, F - (|R| - 1)))`. The same holds for a continuation, with R = the
    ready list plus the node.
  - **Grant.** When `finish()` leaves free threads and an empty ready list, all free threads go to the running
    node with the highest priority. Its atomic is increased under the lock.
  - **Return.** `finish(node)` returns the node's current atomic share, not the `int` it started with.
- `work()`'s `run(node, blas_threads, tid)` signature stays. `blas_threads` is the start share.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST(TopScheduler, PriorityPicksHighestReadyFirst) {
  // parent {4,4,4,4,-1}, priority {1,5,3,2,9}, threads 1, one worker: completed() leaves in
  // order 1, 2, 3, 0, then 4.
}
TEST(TopScheduler, StartShareFollowsPriority) {
  // parent {3,3,3,-1}, priority {6,1,1,8}, threads 8, one worker: the first node run (0) gets
  // blas_threads == 6 (floor(8*6/8) = 6, cap 8 - 2 = 6).
}
TEST(TopScheduler, FreedThreadsGrowTheRunningFront) {
  // parent {2,2,-1}, priority {10,1,11}, threads 8, two workers. Node 0 starts with 7
  // (floor(8*10/11) = 7, cap 7), node 1 with 1. Node 1 returns at once; node 0's run waits
  // (bounded, 2 s) until share(0) == 8 and records it. Expect 8.
}
TEST(TopScheduler, GrantRacingFinishKeepsBudget) {
  // 200 random trees (as RandomTreesStress) with random priorities, threads 8, 8 workers:
  // in every run, shares_in_use() <= 8; after work() returns, shares_in_use() == 0.
}
```

- [ ] **Step 2:** Run `test_top_scheduler`. Expected: FAIL to compile.
- [ ] **Step 3:** Implement. With an empty priority vector, the existing tests must pass unchanged.
- [ ] **Step 4:** Run `test_top_scheduler` locally, then `rtest.sh $WT/build-mtt "TopScheduler|TreeParallel"`.
  Expected: all pass, exit 0.
- [ ] **Step 5:** Commit: `perf: top fronts by remaining path cost, shares that grow (8T engine Part 7)`.

### Task 5: BLAS calls that follow the share, and the priorities in the engine

**Files:**
- Modify: `solver/include/smf/blas_thread_guard.hpp`, `solver/src/parallel/blas_thread_guard.cpp`.
- Modify: `solver/src/dense_kernel_lapack.cpp` (the BLAS-3 wrappers).
- Modify: `solver/src/multifrontal_serial.cpp` (top phase), `solver/include/smf/control.hpp`,
  `solver/benchmarks/bench_replay.cpp`.
- Test: `solver/tests/test_blas_thread_guard.cpp`, `solver/tests/test_tree_parallel.cpp`.

**Interfaces:**
- Consumes Task 4's `TopScheduler(parent, threads, priority)` and `share(node)`.
- Produces:

```cpp
// RAII: while it lives, blas_follow_share() on this thread sets the BLAS thread count (MKL local)
// to *share when it differs from the last value set, and counts it in blas_threads_active().
// null: no effect. Lives inside a BlasThreadGuard, whose destructor restores the setting.
class BlasShareScope {
public:
  explicit BlasShareScope(const std::atomic<int> *share) noexcept;
  ~BlasShareScope() noexcept;
};
void blas_follow_share() noexcept;
```

- `blas_follow_share()` is called first thing in these wrappers:
  - `smf_dgemm`, `smf_dgemm_nt`, `smf_dgemm_tn`;
  - `smf_dgemmt_lower_nt_sub`, `smf_dgemmt_lower_nn_sub`;
  - `smf_dsyrk_lower`;
  - `smf_dtrsm_right_lower_transpose`, `smf_dtrsm_left_lower`.

  Without a scope it costs one thread-local load.
- `Control`: `bool top_critical_shares = false;`. Environment: `SMF_BENCH_TOP_CRITICAL_SHARES`.
- **Priority of a top front.** `cost(s) + priority(parent)`, computed parents first, where
  `cost(s) = f^2 p + ctrl.thread_cost_per_front_entry * f^2`.
  - f and p are the realized sizes (`ws.realized_rows` / `ws.realized_pivots`) when
    `ws.realized_valid && ws.realized_analysis == keep.serial`, else the symbolic `front_size` / `width`.
  - The values are kept in `TreeParallelBuffers::Partition::top_priority`.
  - They are rebuilt when the partition is rebuilt, or when the plan re-plans from realized sizes
    (`replan_realized`).
- In the run lambda: `BlasShareScope scope(ctrl.top_critical_shares && !det ? &sched.share(local) : nullptr);`
  inside the `BlasThreadGuard`.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST(BlasShare, FollowsAGrowingShare) {
  // GTEST_SKIP unless a BlasThreadGuard(4) gives blas_max_threads() == 4 (threaded MKL).
  // BlasThreadGuard g(2); std::atomic<int> a{2}; BlasShareScope s(&a);
  // blas_follow_share(): blas_max_threads() == 2. a = 5; blas_follow_share(): == 5 and
  // blas_threads_active() == 5. After both scopes end: blas_max_threads() is what it was before g.
}
TEST(TreeParallel, CriticalSharesSolveCorrectly) {
  // grid_kkt(200, 7) and opf_like_kkt(20000, 8000, 3), 8 threads, non-deterministic.
  // top_critical_shares on vs off: inertia equal, Info::top_part_fronts equal, the relative backward
  // error of the solve at most 1e-13 in both.
}
```

- [ ] **Step 2:** Run them. Expected: FAIL to compile.
- [ ] **Step 3:** Implement the scope, the wrapper calls, the priority computation and cache, and the flag.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "BlasShare|BlasThreadGuard|TreeParallel|ParallelDeterminism|TopScheduler"`.
  Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: BLAS follows the front's growing share; priorities from path cost (8T engine Part 7)`.

### Task 6: Part 7 A/B and PR

- [ ] **Step 1:** Run the 8-case A/B with `SMF_BENCH_TOP_CRITICAL_SHARES=0` / `1` (Part 5 on in both), then the phase
  split with it on, out `e8t_p7_phase`.
  - Expected: on acopf_grid_160 at 8T, `sum:top_defer + sum:top_schur` at most 0.55x of Task 0's. The spec's target
    is about 6x on the critical path instead of about 2x.
  - Expected: no case's refactor median more than 1% slower.
  - Below target: ledger it and continue. The stop rule judges Parts 5-7 together (Task 10).
- [ ] **Step 2:** Run the board at 16T only (`--threads 16`), Part 7 on vs off, out `e8t_p7_board16`. Expected: track
  geomean at most 1.005. Freed threads now go to running fronts, and the E-cores are where that can hurt.
- [ ] **Step 3:** Switch `top_critical_shares` on. Run `rtest.sh` on `build-mtt` and `build-seq`. Expected: both
  exit 0.
- [ ] **Step 4:** Push `perf/8t-p7-shares` and open the PR against `perf/8t-p5-assembly`. Update `STATUS.md`.

### Part 6: lookahead in the TPP kernel (branch `perf/8t-p6-lookahead`, from `perf/8t-p7-shares`)

### Task 7: the lookahead kernel

**Files:**
- Create: `solver/src/ldlt_lookahead.hpp`.
- Modify: `solver/src/ldlt_kernel.cpp`. The implementation lives here, next to the helpers in its anonymous
  namespace.
- Test: `solver/tests/test_ldlt_lookahead.cpp` (add to `solver/tests/CMakeLists.txt`).

**Interfaces:**
- Consumes `TileRunner`, `TileDag` and `TileHelpers` (`parallel/tile_tasks.hpp`), and `BlasThreadGuard`.
- Produces in `ldlt_lookahead.hpp`:

```cpp
namespace smf {
class TileHelpers;
// Doubles of `work` the lookahead kernel needs: two panel buffers.
inline std::size_t ldlt_lookahead_work_size(int f, const LdltKernelParams &prm) {
  return static_cast<std::size_t>(f) * static_cast<std::size_t>(2 * prm.nb + 2);
}
// ldlt_factor_front (first = 0) with panel lookahead on nthreads threads (helpers, or the runner's
// own threads when null). Only a plain threshold front (!posdef, static_pivot == 0,
// min_abs_pivot == 0, f >= blocked_min_front, nb > 1, nthreads >= 2) takes the lookahead; any other
// calls ldlt_factor_front. Output contract as ldlt_factor_front's; the Schur complement and L differ
// by rounding only.
LdltKernelResult ldlt_factor_front_lookahead(double *F, int f, int num_elig, Int *rows,
                                             const LdltKernelParams &prm, int8_t *tags, double *work,
                                             int nthreads, TileHelpers *helpers);
}  // namespace smf
```

**The algorithm.** It replaces the first loop (plain threshold passes) of the blocked path while no panel stalls.
- **Panels.** Panel k covers `[k*nb, min(num_elig, (k+1)*nb))`. Its W is buffer `k % 2` of `work`.
- **Tasks.** Each panel has three. Task ids are `3k`, `3k+1` and `3k+2`.
  - `P_k` searches and applies panel k's pivots, with BLAS on 1 thread. It is urgent.
  - `N_k` applies panel k's deferred update to panel k+1's columns, every row. It runs on BLAS 1 thread and is
    urgent. The update uses a column-range variant of `deferred_update`: lower GEMM on the square, GEMM below it.
  - `R_k` applies panel k's deferred update to the columns past panel k+1, then panel k's Schur chunk
    (`schur_update` over `[k*nb, end of panel k)`). It runs on BLAS `max(1, nthreads - 1)` threads.
- **Dependencies:**
  - `N_k` follows `P_k` and `R_{k-1}`;
  - `P_{k+1}` follows `N_k`;
  - `R_k` follows `P_k` and `R_{k-1}`.

  `P_{k+1}` overwrites the W buffer `R_{k-1}` read. `N_k` follows `R_{k-1}`, so that is safe.
- **Stall.** When `P_k` stalls, every later task runs as a no-op, so `TileRunner`'s total still completes. After the
  run, the caller continues today's code from that panel:
  - `deferred_update(F, f, ps_k, k, pl_k, num_elig, W_k)`;
  - then the pair / move handling, the rest of the first loop, and the second loop.

  These run with W = buffer 0. The final `schur_update` covers only the pivots that no chunk took.
- **Run.** One `TileRunner::run` per front. Plan decision 3 is why no interchanges need recording.

- [ ] **Step 1: Write the failing tests** in `test_ldlt_lookahead.cpp`.
  - Matrices:
    - `random_indefinite(f = 400, num_elig = 300)` (copy the generator from `test_ldlt_aptp.cpp`);
    - a saddle point, f = 384, with a zero (2,2) block of 128 (2x2 pivots);
    - the `LdltDense.StalledPanelsPullInPartners` pattern grown to f = 320.
  - Settings: u = 0.01 and u = 1e-8, nb = 32. The reference is `ldlt_factor_front` on a copy.

```cpp
TEST(LdltLookahead, SamePivotsAsTpp) {
  // nthreads 2, 4, 8 (helpers == nullptr): rows, tags, num_accepted and inertia equal the reference;
  // max |F - F_ref| over columns [0, p) and over the lower Schur block F[num_elig:f, num_elig:f]
  // at most 1e-12 * max |F_ref|.
}
TEST(LdltLookahead, StallTakesTodaysPath) {
  // f = 256, num_elig = 192, columns [64, 96) with a zero diagonal and their partners in [160, 192):
  // panel 2 stalls. Same checks as above, at nthreads 4.
}
TEST(LdltLookahead, EdgeShapes) {
  // Same checks at nthreads 4 for: num_elig = 301; num_elig == f = 200 (no Schur rows);
  // num_elig = 40 (one panel and a part); a leading 32 x 32 zero block (first panel stalls);
  // an all-zero eligible block (num_accepted == 0, every column delayed).
}
TEST(LdltLookahead, RepeatedRunsAgree) {
  // The saddle point at nthreads 2 (R_k on 1 BLAS thread, so the run is deterministic): 20 runs give
  // bitwise identical F, rows and tags. Catches a race between P_{k+1} and R_k (plan decision 3).
}
```

- [ ] **Step 2:** Run `test_ldlt_lookahead`. Expected: FAIL to compile.
- [ ] **Step 3:** Implement the kernel.
- [ ] **Step 4:** Run `test_ldlt_lookahead` locally, then `rtest.sh $WT/build-mtt "LdltLookahead|LdltDense|LdltAptp"`.
  Expected: all pass, exit 0.
- [ ] **Step 5:** Commit: `perf: TPP kernel with panel lookahead (8T engine Part 6)`.

### Task 8: lookahead in `process_front`, determinism with every part on

**Files:**
- Modify: `solver/src/multifrontal_serial.cpp` (`process_front`'s TPP call and `front_work_need`).
- Modify: `control.hpp`, `info.hpp`, `bench_replay.cpp`.
- Test: `solver/tests/test_tree_parallel.cpp`, `solver/tests/test_parallel_determinism.cpp`.

**Interfaces:**
- Consumes Task 7's `ldlt_factor_front_lookahead` and `ldlt_lookahead_work_size`, and Task 2's `FactorContext::det`.
- Produces:
  - `Control`: `bool tpp_lookahead = false;` and `int tpp_lookahead_min_front = 1024;`.
  - Environment: `SMF_BENCH_TPP_LOOKAHEAD` and `SMF_BENCH_TPP_LOOKAHEAD_MIN`.
  - `Info`: `int lookahead_fronts = 0;`.
- A front takes the lookahead when all of these hold:
  - `ctrl.tpp_lookahead` is on;
  - `tc != nullptr` and `tc->nthreads >= 2`;
  - `!cx.det`;
  - `aprm == nullptr`;
  - it is not small and not a LAPACK front;
  - `f >= ctrl.tpp_lookahead_min_front`;
  - `*prm` is plain threshold.

  `front_work_need` returns `max(tpp, ldlt_lookahead_work_size(...))` for such fronts.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST(TreeParallel, LookaheadSolvesLikeTpp) {
  // opf_like_kkt(20000, 8000, 3) and grid_kkt(200, 7), 8 threads, non-deterministic, min front 128.
  // On vs off: inertia equal, Info::delayed_pivots equal, relative backward error at most 1e-13,
  // Info::lookahead_fronts > 0 when on.
}
TEST(TreeParallel, StopWithStage2PartsOn) {
  // opf_like_kkt(20000, 8000, 3), 8 threads, Parts 5-7 on, Part 6 min front 128, an inertia limit
  // below the true negative count: FactorStatus::InertiaExceeded (no hang: the test's own 60 s
  // watchdog). Then the same FactorKeep without the limit: Success and backward error <= 1e-13.
}
TEST(ParallelDeterminism, Engine8tStage2PartsOn) {
  // As Engine8tStage1PartsOn plus parallel_front_assembly, top_critical_shares and tpp_lookahead on
  // (lookahead never runs in deterministic mode): 1 vs 8 threads bitwise.
}
```

- [ ] **Step 2:** Run them. Expected: FAIL to compile.
- [ ] **Step 3:** Implement the flag, the env reads, the condition, the work size and the counter.
- [ ] **Step 4:** Run the full `rtest.sh $WT/build-mtt`. Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: large top fronts take the lookahead kernel (8T engine Part 6)`.

### Task 9: Part 6 sweep, A/B and PR

- [ ] **Step 1:** Run the 8-case A/B with `off`, `m512`, `m1024` and `m2048` (`SMF_BENCH_TPP_LOOKAHEAD[_MIN]`), with
  Parts 5 and 7 on. Pick the min as in Task 3, and ledger it.
- [ ] **Step 2:** Run the phase split with it on, out `e8t_p6_phase`.
  - Expected: `sum:top_panel` on acopf_grid_160 at 8T at most 0.5x of Task 0's (spec target: visible panel
    6.9 -> about 2.5 ms).
  - Expected: `mf_p2->mf_p3` down.
  - Expected: no case more than 1% slower.
- [ ] **Step 3:** Switch `tpp_lookahead` on. Run `rtest.sh` on `build-mtt` and `build-seq`. Expected: exit 0.
- [ ] **Step 4:** Push `perf/8t-p6-lookahead` and open the PR against `perf/8t-p7-shares`. Update `STATUS.md`.

### Task 10: Stage 2 gate

Run this on the tip of `perf/8t-p6-lookahead`, or on `main` if Daniel has merged the stack.

- [ ] **Step 1: Stop rule.** Compare the acopf_grid_160 8T `mf_p2->mf_p3` median against Task 0's.
  - At most 0.70x: continue.
  - Above 0.70x: stop, report the numbers and re-plan with Daniel. This is the spec's stop rule.
- [ ] **Step 2: Projection.**
  - Re-measure PARDISO and the tip on the same day: out `e8t_gate2`, the command as in Stage 1's Task 17 Step 1.
  - Run `project_run.py --smf e8t_gate2 --rivals e8t_gate2 --threads 8 --gate stage2`.
  - Pass: exit 0, i.e. all 8 cases below 1.00.
- [ ] **Step 3: Board.** Run the board (`e8t_board_gate2`, 1/8/16T, 3 trials + 1 warmup), then
  `ab_compare.py --a e8t_board_gate1 --b e8t_board_gate2 --metric wall`.
  - Pass:
    - each track's geomean at most 1.005;
    - no case at 1.05 or worse on a 3-round rerun;
    - 0 wrong inertia (`score.py --inertia-ref benchmarks/manifests/fronttier_inertia.json`);
    - validation outcomes identical;
    - no backward error more than 2x worse on a case that passes at Stage 1's tip.
  - Also report the ratios against `e8t_board_base` (before Stage 1).
- [ ] **Step 4: Peak RSS.** Pass: the median per case in `e8t_gate2` is at most the one in `e8t_gate1`, at 8T (plan
  decision 4). Also report against `e8t_base`.
- [ ] **Step 5: IPOPT end-to-end** on the 8 cases with the plugin (the PR #64 r2 configuration,
  `--track-override ipopt-plugin`), at the Stage 2 tip and at `main`.
  - Pass: same termination.
  - Pass: identical iteration counts, or each difference traced to rounding, i.e. the same run at 1T reproduces it.
- [ ] **Step 6:** Write `docs/performance/2026-11-XX-8t-engine-stage2-gate.md` (the date of the run). It has:
  - the tables;
  - the verdict;
  - each part's kernel sums;
  - the open items carried from Stage 1 (peak RSS on small cases, cold first factorization).
- [ ] **Step 7: Decide.**
  - **Pass:** report to Daniel. Piece 1 of "best sparse solver first" is done, so propose the merge order of
    #65-#69 and the Stage 2 PRs.
  - **Fail:** report the numbers and re-plan with Daniel.
  - Either way, update `STATUS.md` and the memory.
