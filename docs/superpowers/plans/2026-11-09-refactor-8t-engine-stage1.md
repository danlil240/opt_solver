# 8-thread refactorization engine, Stage 1: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** smf's 8-thread refactorization stops losing to MKL PARDISO on six of the eight PR #64 cases. Those six are
case1354, case2000, portfolio_20k, obstacle_surface_200, pde_control_120 and pde_control_250. The gate for this is
the Stage 1 gate below.

**Architecture:** Stage 1 removes the serial work around the tree. It has four parts:
1. a thread planner priced in time (Part 1);
2. the factor written in place in per-thread slices of one reserved range, with no gather copy (Part 2);
3. solve preparation done per front or cached (Part 3);
4. parallel value preparation and per-analysis setup reuse (Part 4).

The parts land in the order 1, 4, 2, 3. Each part is its own stacked PR, and each has its own on/off control.

**Tech stack:**
- C++20 with MSVC 2022, the threaded MKL build (`SMF_MKL_THREADING=intel`), and GoogleTest.
- Python 3 for the benchmark tools, under pytest.
- The 12900K for every test suite and benchmark, through the scripts in `~/.claude/smf-remote/`.

**Spec:** `docs/superpowers/specs/2026-11-09-refactor-8t-engine-design.md`. Executors read both documents.

**Out of this plan:** Stage 2 (Parts 5-7). Its plan is written after the Stage 1 gate (Task 17), because the spec's
stop rule may change it.

## Global constraints

These come verbatim from the spec. Every task implicitly includes them.

- No loss at 1 or 16 threads. No loss of accuracy.
- Accurate threshold pivoting stays the default, and there is no static pivoting by default.
- Deterministic mode keeps working: its results don't depend on the thread count.
- CPU only and MIT licence. No new third-party dependency.
- Every part does nothing at 1 thread, except where noted.
- Every part has a `Control` flag and an `SMF_BENCH_*` environment override, read in `bench_replay.cpp` next to the
  existing `SMF_BENCH_*` reads (around line 650). A flag defaults to on only after its part's A/B, and stays as the
  off-switch.
- The delay-budget regions keep the 2e7 work floor, because they decide numerics. The opt-in path is
  `delay_budget >= 0`.
- Benchmarks and test suites run on the 12900K only, one at a time, after checking the machine is idle (the scripts
  refuse when it is busy).
- Builds stay local, at below-normal priority, with at most 6 jobs.
- Never commit or copy the licensed CoinHSL (`build-deps/coinhsl-*`, `C:\smfkit\coinhsl_bin`) or the Mittelmann
  `.mod`/`.nl` files.
- Update `STATUS.md` (header, affected section, dated changelog line) after every PR, gate and plan change.

## Review focus

These are inputs the spec implies but no task's main test exercises. Each line names the test that pins it and the
task that owns it.

1. **Refactorizing into the same `FactorKeep` after the thread count changes.** This happens in the IPOPT plugin
   with `SMF_NUM_THREADS`, or in a session going 8 -> 2 -> 1 -> 8 threads. The store must re-reserve, and every
   solve must stay correct.
   Test: `FactorStoreEngine.RefactorAcrossThreadCounts`, Task 10.
2. **A factorization that aborts.** The cases are `InertiaExceeded` under an inertia limit, and the work limit of
   scale-on-delay's early stop. The next factorization must be correct and must not reuse a stale plan or layout.
   Tests: `TreeParallel.PlanCacheSurvivesAbort` (Task 2) and `TreeParallel.LayoutCacheClearedAfterAbort` (Task 14).
3. **Delays that suddenly grow the factor past a slice's reservation.** Example: `cascade_kkt` values with spread
   after values without. Expect the regather fallback, a correct solve, and the fallback counted in
   `Info::factor_store_regathers`.
   Test: `FactorStoreEngine.OverflowRegathers`, Task 10.
4. **Tiny and empty matrices at 8 threads:** n = 0, n = 1, and a single supernode. Expect no reservation failures,
   a plan of 1 thread, and no parallel value prep.
   Tests: `FactorStore.ZeroAndTinySlices` (Task 8) and `FactorStoreEngine.TinyAndEmptyMatrices` (Task 10).
5. **Deterministic mode with every Stage 1 part on, at 1 versus 8 threads.** Expect bit-for-bit identical
   solutions.
   Test: `ParallelDeterminism.Engine8tStage1PartsOn`, Task 16.

## Conventions used by every task

**Worktree and branches.**
- Worktree: `C:\Users\daniel\projects\opt_solver-wt\engine8t`. `WT` below means that path.
- Branch `feat/engine-8t` holds the spec, this plan and the harness (Task 0). Its PR is "8T engine: spec, plan,
  harness".
- Each part branches from the previous one: `perf/8t-p1-planner`, `perf/8t-p4-valueprep`, `perf/8t-p2-store`,
  `perf/8t-p3-layout`.
- Each part's PR targets the previous branch, and Daniel merges them in order. Create a branch with
  `git -C $WT switch -c <branch>`.

**Build.** `cmd //c "%USERPROFILE%\.claude\smf-remote\build_mtt.cmd" <worktree> [target]`, created in Task 0. It
configures `build-mtt` (threaded MKL, tests and benchmarks on, `SMF_BENCH_CUDSS=OFF`) and builds at below-normal
priority with `--parallel 6`. Expected last line: `ALL_OK`.

**Tests.**
- `bash ~/.claude/smf-remote/rtest.sh $WT/build-mtt "<regex>"`. Exit 0 means pass.
- Run it without `MSYS_NO_PATHCONV`.
- A quick local run of one pure-logic gtest is allowed: `$WT/build-mtt/tests/Release/<exe> --gtest_filter=<name>`.

**8-case A/B.**

```bash
RRUN_SRC=$WT bash ~/.claude/smf-remote/rab.sh e8t $WT/build-mtt/benchmarks/Release/bench_replay.exe <out> - -- \
  --manifest benchmarks/manifests/probe_refactor8t.json --threads 8 --rounds 3 --seq-steps all --timeout 300 \
  --variant off=SMF_BENCH_<FLAG>=0 --variant on=SMF_BENCH_<FLAG>=1
python $WT/tools/bench/ab_compare.py --a benchmarks/results/remote/<out>/off --b benchmarks/results/remote/<out>/on \
  --manifest $WT/benchmarks/manifests/probe_refactor8t.json --metric refactor
python $WT/tools/bench/project_run.py --smf benchmarks/results/remote/<out>/on \
  --rivals benchmarks/results/remote/e8t_base --manifest $WT/benchmarks/manifests/probe_refactor8t.json --threads 8
```

**Phase split.** Run the 8 cases with the phase trace on:

```bash
bash ~/.claude/smf-remote/rpy.sh <out> "SMF_BENCH_PHASE_TRACE=1 <flags>" tools/bench/phase_run.py <tag> -- --threads 1,8
```

Then tabulate with `python $WT/tools/bench/phase_split.py benchmarks/results/remote/<out>`.

**Results** land in `C:\Users\daniel\projects\opt_solver\benchmarks\results\remote\<out>`. That directory is not
committed.

---

### Task 0: harness, baselines and tooling (branch `feat/engine-8t`)

**Files:**
- Cherry-pick 9159a58 from `probe/refactor-8t`. It brings `tools/bench/run.py --seq-steps`,
  `benchmarks/manifests/probe_refactor8t.json`, `tools/bench/test_score.py` and
  `docs/performance/2026-11-08-refactor-8t-probe.md`.
- Modify: `docs/performance/2026-11-08-refactor-8t-probe.md` (the PGLib correction).
- Modify: `tools/bench/ab_run.py`, `tools/bench/ab_compare.py`.
- Create: `tools/bench/project_run.py`, `tools/bench/test_project_run.py`.
- Create: `solver/src/phase_trace.hpp`, `solver/src/phase_trace.cpp`, and add them to the `smf` sources in
  `solver/CMakeLists.txt`.
- Modify: `solver/src/factor_indef.cpp`, `solver/src/multifrontal_serial.cpp` (marks only).
- Create: `tools/bench/phase_run.py`, `tools/bench/phase_split.py`, `tools/bench/test_phase_split.py`.
- Outside the repo: `~/.claude/smf-remote/build_mtt.cmd`, `~/.claude/smf-remote/rpy.sh`, and a README section.

**Interfaces:**
- Produces `project_run.project(records: list[dict], factorizations: dict[str, int]) -> dict[tuple[str, int, str], float]`.
  - Keys are `(matrix_id, threads, solver_id)`.
  - The value is analyse + factor + max(0, N - batch_count) x refactor, each a median over the non-warmup records.
    This is the formula of the session script `probe8t_an.py`.
- Produces `project_run.ratios(proj, threads: int, rival: str = "mkl_pardiso") -> dict[str, float]`, giving
  smf / rival per case.
- Produces the gate sets `STAGE1_WINS` (the six cases above) and `STAGE2_WINS` (all 8). `--gate stage1|stage2` exits
  1 if any case in the set has a ratio of 1.00 or more, or if any smf record is not `outcome == "Success"`.
- Produces `smf::phase_mark(const char *name) noexcept` in `phase_trace.hpp`.
  - It prints `# seg <name> t=<seconds, %.7f, steady clock since process start>` to stderr, and only when
    `SMF_BENCH_PHASE_TRACE=1`. The environment is read once and cached in a static.
  - `smf::phase_trace_on() noexcept` returns that flag.
- Produces `phase_split.split(lines: list[str]) -> list[dict[str, float]]`: one dict per factorization (a
  factorization starts at `fi_entry`), mapping segment name `"<a>-><b>"` to milliseconds.
  `phase_split.table(dir)` prints median ms per case, thread count and segment.
- Produces `ab_run.py --seq-steps all` (passed through to run.py) and `ab_compare.py --metric refactor` (median
  `refactor_ms`).

**Mark names**, in order. They are placed where the probe branch `probe/front-trace-8t` placed `probe_mark`, plus
the layout sub-steps:

| mark | where |
|---|---|
| `fi_entry` | entry of `factor_indef` |
| `fi_mapped` | after `map_values_to_permuted` |
| `fi_scaled` | after the scale loop |
| `fi_call` | before `multifrontal_factor_serial` |
| `mf_setup` | before `phase1_start` |
| `mf_p2` | before phase 2 |
| `mf_p3` | before phase 3 |
| `mf_gathered` | after `gather_seconds` |
| `mf_steps` | after the per-thread step loop |
| `mf_diag` | after the `diag_entries` block |
| `mf_top` | after the `det` / top-tree block |
| `mf_toprows` | after `solve_top_rows` and `plan_solve_threads` |
| `mf_ret` | after `finish_solve_layout` |
| `fi_end` | after `++ws.warmup_calls` |

The serial path emits only the marks it reaches.

- [ ] **Step 1:** Cherry-pick 9159a58, then fix the doc's PGLib text.
  - Replace "PGLib does not scale at all ... carry 14-21k delayed pivots each" with: PGLib runs on one thread at 8T
    because `kTreeParallelMinWork = 2e7` is a hard floor and the planner prices flops only. Forced to 8 threads it
    gets 2.2x with identical inertia (trace 2026-11-09). Point to the spec.
  - Replace the verdict's item 2 accordingly.
  - Commit: `bench: run.py --seq-steps and 8T probe manifest (from #64), PGLib correction`.
- [ ] **Step 2: Write the failing tests in `tools/bench/test_project_run.py`.**

```python
def test_projection_formula():
    recs = [rec("c", 8, "smf", analyse_ms=10, factor_ms=50, refactor_ms=4, batch_count=5),
            rec("c", 8, "smf", analyse_ms=12, factor_ms=54, refactor_ms=6, batch_count=5)]
    p = project_run.project(recs, {"c": 25})
    assert p[("c", 8, "smf")] == 11 + 52 + 20 * 5

def test_warmups_ignored_and_gate():
    recs = [rec("a", 8, "smf", 1, 1, 1, 1), rec("a", 8, "mkl_pardiso", 1, 2, 2, 1),
            rec("a", 8, "smf", 100, 100, 100, 1, warmup=True)]
    r = project_run.ratios(project_run.project(recs, {"a": 3}), 8)
    assert r["a"] == pytest.approx(4 / 7)
    assert project_run.gate_failures(r, {"a"}, recs) == []
    assert project_run.gate_failures({"a": 1.0}, {"a"}, recs) == ["a"]
```

  Helper `rec(...)` builds a record dict with the fields `matrix_id threads solver_id analyse_ms factor_ms
  refactor_ms batch_count outcome warmup`.
- [ ] **Step 3:** Run `python -m pytest $WT/tools/bench/test_project_run.py -q`. Expected: it fails (no module).
- [ ] **Step 4:** Implement `project_run.py`: `project`, `ratios`, `gate_failures`, and a CLI with `--smf DIR`,
  `--rivals DIR`, `--manifest`, `--threads`, `--gate`.
  - The CLI prints a table per case: N, refactor ms, projected ms, rival ms, ratio, median `peak_rss_bytes` in MB.
  - N comes from the manifest's `factorizations`.
- [ ] **Step 5:** Run the pytest command again. Expected: PASS.
- [ ] **Step 6:** Add `--seq-steps` to `ab_run.py` (passed to every run.py call) and `refactor` to `ab_compare.py`'s
  `--metric` choices.
  - Add one test to the existing `tools/bench/test_track_override.py` asserting that ab_run's run.py command contains
    `--seq-steps all` when it is given.
  - Run `python -m pytest $WT/tools/bench -q`. Expected: all pass.
- [ ] **Step 7: Write `test_phase_split.py`.** It uses a captured stderr with two factorizations of marks.
  - Assert `split(lines)[1]["fi_entry->fi_mapped"] == pytest.approx(0.25)`.
  - Assert that a factorization missing `mf_p2` still gives its other segments.
  - Run it: FAIL. Implement `phase_split.py`. Run it: PASS.
- [ ] **Step 8:** Implement `phase_trace.hpp/.cpp` and place the marks.
  - `phase_run.py` is `ftrace_run.py`'s timing mode from `probe/front-trace-8t:tools/probe/ftrace_run.py`, without
    the per-front trace. Its arguments: `<binary> <manifest> <out> [--trials 3] [--threads 1,8] [--cases ...]`. Each
    run's stderr goes to `<out>/phase_<case>_T<t>_<trial>.txt`.
  - Build with `build_mtt.cmd $WT bench_replay`.
- [ ] **Step 9: Create the out-of-repo helpers.**
  - `build_mtt.cmd`: the session's `ftrace8t/build.cmd`, taking `%1` = worktree, `%2` = target (default all), and
    `%3` = MKL threading (default `intel`; `sequential` builds `build-seq`).
  - `rpy.sh`: the session's `ftrace8t/rprobe.sh`, generalized. Its arguments: `<out> "<VAR=v ...>|-" <script>
    <tag> [--manifest <repo-relative manifest>] -- <args>`.
    - `<script>` is a local path, or a bare name under `$WT/tools/bench`.
    - It copies the script and the manifest (default `benchmarks/manifests/probe_refactor8t.json`) into the kit.
    - It runs `python <script> build-<tag>\benchmarks\Release\bench_replay.exe <manifest> C:\smfkit\runs\<out>
      <args>`. The binary for `build-<tag>` is shipped first by the copy step `rrun.sh` uses (factor that into a
      function both scripts share).
    - Build the `.cmd` with the heredoc plus `sed 's/$/\r/'` (never printf with Windows paths), check for
      `LAUNCHED`, and fetch to `results/remote/<out>`.
  - Add a README section for both.
- [ ] **Step 10:** Check that the phase trace changes nothing when off.
  - Run `rtest.sh $WT/build-mtt` (full suite). Expected: exit 0.
  - Then run `rpy.sh e8t_phase_base "SMF_BENCH_PHASE_TRACE=1" phase_run.py mtt -- --threads 1,8`.
  - Expected: `phase_split.py` reproduces the 2026-11-09 serial segments within 15%. For example, pde_control_250
    8T is about 2.3 ms from `fi_mapped` to `fi_scaled`, and the gather is about 2.4 ms.
- [ ] **Step 11: Baselines at the branch base**, sequentially, each after checking the machine is idle:
  - `rrun.sh e8t <bench_replay> e8t_base - -- --manifest benchmarks/manifests/probe_refactor8t.json --threads 1,8
    --trials 3 --warmups 1 --allow-short-trials --timeout 300 --seq-steps all --solvers smf,mkl_pardiso`;
  - `rrun.sh e8t <bench_replay> e8t_board_base - -- --manifest benchmarks/manifests/fronttier.json --threads 1,8,16
    --trials 3 --warmups 1 --allow-short-trials --timeout 300 --solvers smf`.

  Check: `project_run.py --smf e8t_base --rivals e8t_base --threads 8` reproduces PR #64's ratios within 5% (for
  example case2000 about 1.67).
- [ ] **Step 12:** Commit (`bench: projection, phase trace and A/B seq-steps for the 8T engine`), push, and open the
  PR. Update `STATUS.md`.

### Part 1: thread planner priced in time (branch `perf/8t-p1-planner`)

### Task 1: front-cost work model and the floor split

**Files:**
- Modify: `solver/src/parallel/thread_plan.hpp` and `.cpp`, `solver/include/smf/control.hpp`,
  `solver/src/multifrontal_serial.cpp` (planning block, about lines 1157-1240), `solver/benchmarks/bench_replay.cpp`.
- Test: `solver/tests/test_thread_plan.cpp`, `solver/tests/test_tree_parallel.cpp`.

**Interfaces:**

Produces in `thread_plan.hpp`:

```cpp
// Floor of the delay-budget regions only (was kTreeParallelMinWork).
inline constexpr double kBudgetRegionMinWork = 2e7;

struct FrontCostModel {
  double per_front_entry = 0.0;           // c_asm: work units per front entry (f^2)
  double per_front = 0.0;                 // c_front: work units per front
  std::span<const Int> realized_rows;     // empty: AnalysisKeep::front_size
  std::span<const Int> realized_pivots;   // empty: supernode width
};

// work[s] = f^2 p + per_front_entry f^2 + per_front (f, p realized when given)
TreeView tree_view(const AnalysisKeep &keep, bool tiles_on, bool blas_parallel,
                   std::vector<Int> &parent, std::vector<double> &work,
                   std::vector<std::uint8_t> &split, int tile_min = kTopTileMinFront,
                   const FrontCostModel &fc = {});
```

- `ThreadCosts` gains `double min_work = kBudgetRegionMinWork;`. `plan_is_serial` and `factor_plan_terms` use
  `c.min_work` (`factor_plan_terms` takes `double min_work` as a new last parameter) instead of the constant.
  `kTreeParallelMinWork` is removed.
- `Control` gains:
  - `bool thread_plan_front_costs = false;`
  - `double thread_cost_per_front_entry = 0.0;`
  - `double thread_cost_per_front = 0.0;`

  Task 3 sets the fitted defaults and turns `thread_plan_front_costs` on.
- Environment: `SMF_BENCH_PLAN_FRONT_COSTS=0|1`.

When `thread_plan_front_costs` is on, the engine:
- builds the plan's view with the `FrontCostModel` from `Control`;
- uses `min_work = 0` for the plan and for `partition_tasks`.

The budget partition (`delay_budget >= 0`) always uses a flops-only view and `kBudgetRegionMinWork`. That view is a
second `tree_view` call into its own `TreeParallelBuffers` arrays (`budget_view_parent`, `budget_view_work`,
`budget_view_split`, `budget_sums`).

- [ ] **Step 1: Write the failing tests** in `test_thread_plan.cpp`.

```cpp
TEST(ThreadPlan, FrontCostsAddAssemblyAndOverhead) {
    // laplacian analysed as in FactorTermsReproducePlan; for every supernode s:
    // view.work[s] == f*f*p + 3.0*f*f + 500.0 with FrontCostModel{3.0, 500.0}
}
TEST(ThreadPlan, ZeroFrontCostsKeepFlopsView)       // FrontCostModel{} gives exactly f*f*p
TEST(ThreadPlan, ManyTinyFrontsParallelWithoutFloor) {
    // Synth: 64 chains of 300 fronts (work 10 + 5e4 each) under one root of work 10 + 5e4
    ThreadCosts c; c.min_work = 0.0;
    EXPECT_FALSE(plan_is_serial(total, c));
    EXPECT_EQ(plan_threads(view, 8, c).threads, 8);
    ThreadCosts floor; // min_work = 2e7: the same tree's flops-only total (64*300*10) stays serial
    EXPECT_TRUE(plan_is_serial(64.0 * 300 * 10, floor));
}
TEST(ThreadPlan, SmallTreeStaysSerialWithoutFloor)  // total 2e6 <= 4 per_thread: plan_is_serial true, threads 1
```

  In `test_tree_parallel.cpp`:

```cpp
TEST(TreeParallel, DelayBudgetRegionsUnchangedByFrontCosts) {
    // cascade_kkt(40, 600, 4.1, 3.0); ctrl.delay_budget = 20; num_threads 4; auto_threads on.
    // Factor with thread_plan_front_costs off, then on (cost_per_front 1e5, per_front_entry 10).
    // EXPECT_EQ on Info::delayed_pivots, static_pivots, inertia; solution vectors bit-for-bit equal.
}
```

- [ ] **Step 2:** Build `test_thread_plan` and `test_tree_parallel` and run the five tests.
  Expected: compile failure (no `FrontCostModel`, no `min_work`).
- [ ] **Step 3:** Implement the interfaces above.
  - `tree_view` uses the realized spans when their size equals ns.
  - Replace the old constant in `factor_plan_terms` (thread_plan.cpp:532), `plan_is_serial` (:602), and in
    `multifrontal_serial.cpp` (`below_min_work` and both `partition(...)` calls).
  - Add the `bench_replay` environment read.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "ThreadPlan|TreeParallel"`. Expected: exit 0, including every
  existing ThreadPlan test.
- [ ] **Step 5:** Commit: `perf: front-cost work model for the thread plan; work floor kept for budget regions only`.

### Task 2: re-plan from realized front sizes

**Files:**
- Modify: `solver/include/smf/workspace.hpp` (`NumericWorkspace`, `ThreadPlanCache`), `solver/src/workspace.cpp`
  (`prepare_for_analysis`), `solver/src/multifrontal_serial.cpp` (`process_front`, planning block, `FrontPass`
  stats).
- Test: `solver/tests/test_tree_parallel.cpp`.

**Interfaces:**
- Consumes: Task 1's `FrontCostModel` and `tree_view`.
- Produces:
  - `NumericWorkspace::realized_rows` and `realized_pivots` (`std::vector<Int>`, size ns, set by
    `prepare_for_analysis`). They are written by `process_front` after the kernel: `f` (rows, delays included) and
    the accepted pivots. Each front writes only its own entry, so there is no race.
  - `ThreadPlanCache` gains `double planned_total = 0.0;` and `bool from_realized = false;`.
  - `FrontPass` stats gain `double realized_work`: each front adds `f^2 p + c_asm f^2 + c_front`, and the per-pass
    sums are added in `add_stats`.
  - `Info` gains `int plan_from_realized = 0;`.

**Re-plan rule**, only with `thread_plan_front_costs` on:
- After a successful factorization, keep the realized total.
- The next factorization of the same analysis re-plans from `realized_rows` / `realized_pivots` when
  |realized_total - planned_total| > 0.10 x planned_total. It then sets `from_realized = true`.
- A failed factorization leaves the cache as it was.

- [ ] **Step 1: Write the failing tests.**

```cpp
TEST(TreeParallel, ReplansWhenDelaysGrowFronts) {
    // cascade_kkt(60, 1500, 4.1, 0.0) analysed; values V0 = no spread (few delays),
    // V1 = cascade_kkt(60, 1500, 4.1, 3.0) values on the same pattern (many delays).
    // ctrl.thread_plan_front_costs = true, cost_per_front 1e5; one NumericWorkspace and FactorKeep.
    // factor V0 -> info.plan_from_realized == 0; factor V1 -> 0 (plan made before V1's sizes were known);
    // factor V1 again -> 1; factor V1 a third time -> 0 (realized total now within 10%).
}
TEST(TreeParallel, PlanCacheSurvivesAbort) {
    // Same matrix; ctrl.max_negative_pivots = 1 forces FactorStatus::InertiaExceeded on V1, then V0
    // (max_negative_pivots = -1) factors with plan_from_realized == 0 and threads_used equal to the first
    // V0 factorization's.
}
```
- [ ] **Step 2:** Run them. Expected: FAIL (no field `plan_from_realized`).
- [ ] **Step 3:** Implement as described in the Interfaces block.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "TreeParallel|RepeatedFactor|IpmSession"`. Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: thread plan re-plans from realized front sizes`.

### Task 3: fit the front costs, refit the thread costs, switch on

**Files:**
- Create: `tools/bench/fit_front_costs.py`, `tools/bench/test_fit_front_costs.py`,
  `benchmarks/manifests/fronttier_ipopt.json` (the `fronttier.json` cases with `track == "ipopt"`).
- Modify: `solver/include/smf/control.hpp` (defaults and their evidence comment),
  `tools/bench/fit_thread_costs.py` (only if its trace parser needs a new field).

**Interfaces:**
- Produces `fit_front_costs.fit(fronts: list[tuple[f, ne, p, seconds]], fprintf_s: float) -> (c_asm, c_front)`.
  - It does a least-squares fit of `t = a f^2 p + b f^2 + c` over the fronts.
  - It returns `c_asm = b / a` and `c_front = (c - fprintf_s) / a`, where `fprintf_s` is the probe's per-front
    print cost.
  - A front's seconds is the next front's `ta` minus this front's `ta`. The last front of a factorization runs to the
    factorization's end.
- Input: the probe trace files `fronts_*_T1_trace_*.txt`. Their columns are
  `phase f ne p me blas tiled lapack flops ta tk0 tk1 s parent psym panel defer schur stall nstall nfull`.

- [ ] **Step 1: Write `test_fit_front_costs.py`.** Use synthetic fronts generated from known `a = 1e-10`,
  `b = 3e-9`, `c = 2e-6` with `fprintf_s = 5e-7`, and assert that `fit` returns `(30.0, 15000.0)` within 1%. Run
  it: FAIL. Implement. Run it: PASS.
- [ ] **Step 2: Collect 1T front traces of the board's ipopt track** with the probe binary already on the 12900K
  (`C:\smfkit\smfbench\build-ftrace`, branch `probe/front-trace-8t`):

  `bash rpy.sh e8t_fronts_ipopt "SMF_PROBE_FRONTS=1 SMF_PROBE_MINF=0" ../opt_solver-wt/ftrace8t/tools/probe/ftrace_run.py ftrace --manifest benchmarks/manifests/fronttier_ipopt.json -- --threads 1 --trials 1`

  - Measure `fprintf_s` as (traced 1T refactor - untraced 1T refactor) / fronts on the 8 cases, using `ftrace8t`'s
    `steps_*_T1_t*.txt` against `*_trace.txt`.
- [ ] **Step 3:** Run `fit_front_costs.py` on the 8 cases' traces in `results/remote/ftrace8t` plus
  `e8t_fronts_ipopt`. Record `c_asm` and `c_front`, and the fit's R², per class.
- [ ] **Step 4: Refit the thread costs.**
  - Set the new defaults in `control.hpp`, with `thread_plan_front_costs = true`.
  - Collect plan traces and truth times with the new binary, the way the header of `fit_thread_costs.py` says:
    `SMF_BENCH_AUTO_THREADS=0` at 1/2/4/8/16 threads and `SMF_BENCH_PLAN_TRACE=1`, on `fronttier.json` and
    `probe_refactor8t.json`, through `rrun.sh`.
  - Refit `thread_cost_per_thread`, `per_top_front`, `per_split_front_thread` and `split_cap` with
    `fit_thread_costs.py`, and set them.
  - Write the evidence into the comment above the defaults: data dirs, R², and the cases that pick a different T than
    before.
- [ ] **Step 5:** Run `rtest.sh $WT/build-mtt` (full suite). Expected: exit 0. If a ThreadPlan or TreeParallel test
  premise (for example "the grid repays more than 2 threads") no longer holds, re-derive the premise from the new
  costs. Never loosen an assertion to pass.
- [ ] **Step 6:** Commit: `perf: fitted front costs; thread plan priced in time by default`.

### Task 4: Part 1 A/B and PR

- [ ] **Step 1:** Run the 8-case A/B (Conventions) with `SMF_BENCH_PLAN_FRONT_COSTS=0` / `1`.
  - Expected: case1354, case2000 and portfolio_20k refactor at 8T at most 2.9, 6.0 and 5.8 ms (the forced-8T times
    plus 5%).
  - Expected projection ratios: about 0.46 / 0.75 / 0.80, and no other case slower than 2%.
- [ ] **Step 2: Board A/B for thread counts.**

  `rab.sh ... -- --manifest benchmarks/manifests/fronttier.json --threads 1,8,16 --rounds 1 --variant off=... --variant on=...`

  Then `ab_compare.py --metric wall`.
  - Pass: each track's geomean on/off at most 1.005.
  - Pass: no case worse than 1.05 on a rerun of that case with `--rounds 3 --cases <id>`.
  - Pass: 1T identical within noise, since the planner does nothing at 1T.
- [ ] **Step 3:** Run `rtest.sh` on `build-mtt` and on `build-seq` (`build_mtt.cmd $WT all sequential`).
  Expected: both exit 0.
- [ ] **Step 4:** Push, open the PR "perf: thread plan priced in time (8T engine Part 1)" against `feat/engine-8t`,
  with the A/B tables in the body. Update `STATUS.md`.

### Part 4: parallel value preparation and setup reuse (branch `perf/8t-p4-valueprep`)

### Task 5: parallel map and scale

**Files:**
- Create: `solver/src/parallel/par_for.hpp`.
- Modify: `solver/src/workspace.cpp` (`map_values_to_permuted`), `solver/include/smf/workspace.hpp` (its
  declaration), `solver/src/factor_indef.cpp` (scale loop, about lines 482-499), `control.hpp`, `bench_replay.cpp`.
- Test: `solver/tests/test_scaled_value_prep.cpp`.

**Interfaces:**
- Produces in `par_for.hpp`:

```cpp
// Runs fn(tid, begin, end) over [0, n) in `threads` contiguous ranges on ws's factor pool
// (ws.ensure_pool(threads).run(...)); threads <= 1 or n < min_n: fn(0, 0, n) on the caller.
template <class F> void parallel_ranges(NumericWorkspace &ws, int threads, std::size_t n,
                                        std::size_t min_n, F &&fn);
```

- `map_values_to_permuted` gains `int threads = 1` as a new last parameter. With `threads > 1`, these become
  parallel:
  - the finiteness and max pass (a per-range flag and max, reduced after);
  - the plain copy `clean_values.assign(uv, uv + nv)`;
  - the one-to-one `orig_to_perm_idx` scatter when `orig_to_perm_idx` is a permutation.

  These stay serial: the `orig_to_clean_idx` / `orig_to_perm_direct` `+=` scatters, which may have duplicates.
- In `factor_indef.cpp`, the `direct` scale-and-scatter loop runs over column ranges, with boundaries chosen so each
  range holds about nnz / threads entries (binary search on `col_ptr`). Each range has its own max.
- `Control` gains `bool parallel_value_prep = false;` and `std::size_t parallel_value_prep_min_entries = 100000;`.
  Environment: `SMF_BENCH_PARALLEL_VALUE_PREP`.
- Thread count: `ws.thread_plan.threads` when the cache holds a plan for this analysis and `ctrl.auto_threads` is
  on; otherwise `ctrl.num_threads`.

- [ ] **Step 1: Write the failing test.**

```cpp
TEST(ScaledValuePrep, ParallelPrepBitwiseEqualsSerial) {
    // opf_like_kkt(3000, 1200, 1) and a 400k-entry laplacian; Scaling Matching and None; orig layout
    // and cleaned layout. For threads in {2, 8}: ws.perm_values, ws.clean_values, fkeep.scale and
    // Info max |a| are == (bitwise, std::memcmp) to threads = 1 with parallel_value_prep_min_entries = 0.
}
```

- [ ] **Step 2:** Run it. Expected: FAIL (no `parallel_value_prep`).
- [ ] **Step 3:** Implement `par_for.hpp`, the `map_values_to_permuted` ranges, the scale ranges and the flag.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "ScaledValuePrep|FactorValuesReuse|RefactorAllocations"`.
  Expected: exit 0. `RefactorAllocations` proves the pool adds no matrix-sized allocation.
- [ ] **Step 5:** Commit: `perf: parallel value mapping and scaling (8T engine Part 4)`.

### Task 6: per-analysis setup reuse

**Files:** `solver/src/factor_indef.cpp` (fkeep setup after the scale loop), `solver/src/multifrontal_serial.cpp`
(setup block, about lines 1085-1130 and 1329-1352), `solver/include/smf/factor_posdef.hpp`
(`FactorKeep::setup_analysis`). Test: `solver/tests/test_repeated_factor.cpp`.

**Interfaces:** `FactorKeep` gains `std::uint64_t setup_analysis = 0;`. It holds the `AnalysisKeep::serial` whose
setup this `FactorKeep` holds, and is 0 after an error path clears it.

- [ ] **Step 1: Measure.**
  - Run `rpy.sh e8t_p4_setup "SMF_BENCH_PHASE_TRACE=1" phase_run.py mtt -- --threads 8` with extra temporary marks
    around each setup statement. Don't commit those marks.
  - List each statement over 0.05 ms on any case. Only the listed statements are candidates.
  - Expected candidates: `fkeep.perm` / `iperm` assign, `num_perm` iota, `num_iperm` copy, `front_pass(ns)`,
    `node_blocks.resize(ns)`, per-thread `b.pos` resets, and `std::vector<FrontPass> tpass`.
- [ ] **Step 2: Write the failing test.**

```cpp
TEST(RepeatedFactor, SetupReuseResetsOnNewAnalysis) {
    // Analyse A (laplacian 120) and B (opf_like_kkt 2000/800); one FactorKeep and workspace:
    // factor A, A, B, A with parallel_value_prep on and num_threads 8; each solve's residual < 1e-12,
    // and after B fk.perm == B's analysis perm, after the last A fk.perm == A's (element-wise).
    // fk.setup_analysis == ak.serial after each success; 0 after a forced IllegalValue (NaN value).
}
```

- [ ] **Step 3:** Run it. Expected: FAIL (no `setup_analysis`).
- [ ] **Step 4:** Skip each candidate when `fk.setup_analysis == keep.serial`.
  - Keep any per-thread buffers that are recreated per call in `TreeParallelBuffers`, sized once per analysis.
  - Set `setup_analysis` on success and clear it on every error return.
- [ ] **Step 5:** Run `rtest.sh $WT/build-mtt "RepeatedFactor|TreeParallel|IpmSession|CApi"`. Expected: exit 0.
- [ ] **Step 6:** Commit: `perf: skip per-refactorization setup unchanged since the last factorization`.

### Task 7: Part 4 A/B and PR

- [ ] **Step 1:** Switch `parallel_value_prep` on by default and run the 8-case A/B with
  `SMF_BENCH_PARALLEL_VALUE_PREP=0` / `1`. Then run the phase split.
  - Expected: `fi_entry->fi_scaled` on acopf_grid_160 under 1.0 ms and on pde_control_250 under 0.7 ms.
  - Expected: setup (`fi_call->mf_setup`) under 0.3 ms on every case.
  - Expected: no case slower.
- [ ] **Step 2:** Run `rtest.sh` on the `build-mtt` and `build-seq` builds. Expected: both exit 0.
- [ ] **Step 3:** Push and open the PR (against `perf/8t-p1-planner`). Update `STATUS.md`.

### Part 2: factor written in place (branch `perf/8t-p2-store`)

### Task 8: `FactorStore` and the virtual-memory layer

**Files:**
- Create: `solver/include/smf/factor_store.hpp`, `solver/src/factor_store.cpp`, `solver/src/vm_reserve.hpp`,
  `solver/src/vm_reserve.cpp`. Add them to the `smf` sources.
- Test: `solver/tests/test_factor_store.cpp`, registered as `add_test(NAME FactorStore ...)` in
  `solver/tests/CMakeLists.txt` like `test_thread_plan`.

**Interfaces:**

```cpp
// vm_reserve.hpp (Windows: VirtualAlloc MEM_RESERVE / MEM_COMMIT / VirtualFree MEM_DECOMMIT|MEM_RELEASE;
// POSIX: mmap PROT_NONE + mprotect, madvise(MADV_DONTNEED) + mprotect PROT_NONE, munmap)
void *vm_reserve(std::size_t bytes) noexcept;          // nullptr on failure
bool vm_commit(void *p, std::size_t bytes) noexcept;   // p and bytes multiples of vm_granule()
void vm_decommit(void *p, std::size_t bytes) noexcept;
void vm_release(void *p, std::size_t bytes) noexcept;
std::size_t vm_granule() noexcept;                     // 64 KiB on Windows, page size elsewhere

template <class T> class FactorSlice {   // one thread's part; size() elements written, never zero-filled
public:
  T *data() noexcept; const T *data() const noexcept;
  std::size_t size() const noexcept;
  bool grow(std::size_t n);             // size() = n; commits in 2 MiB chunks; beyond the reservation:
                                        // moves to a private reservation of 2n (copying size()), detached() = true
  bool detached() const noexcept;
};

template <class T> class FactorStore {
public:
  bool reserve_slices(int slices, std::size_t slice_capacity);  // keeps the reservation when unchanged
  int slices() const noexcept;
  FactorSlice<T> &slice(int t);
  std::size_t slice_offset(int t) const noexcept;   // elements from data(); valid while attached
  bool all_attached() const noexcept;
  void regather();          // detached slices: one contiguous slice of sum(size), copied in slice order;
                            // afterwards slices() == 1 and slice_offset(0) == 0
  T *data() noexcept; const T *data() const noexcept;
  std::size_t size() const noexcept;   // extent: end of the highest used element
  std::size_t used() const noexcept;   // sum of slice sizes
  bool empty() const noexcept;         // used() == 0
  void clear() noexcept;               // every slice to size 0, reservation kept
  void assign(std::size_t n, const T &v);              // one slice of n copies (legacy paths)
  template <class F> void for_each_used(F &&f) const;  // f(const T *p, std::size_t n) per slice
  void trim();   // per slice: committed - used > max(used / 4, 16 MiB) -> decommit down to used rounded up
  std::size_t committed_bytes() const noexcept;  // for the trim rule and its test
  // movable; copy keeps every slice offset (copies used ranges only)
};
```

- [ ] **Step 1: Write the failing tests.**

```cpp
TEST(FactorStore, SlicesAreDisjointAndAddressedFromBase)  // 4 slices, cap 1<<20; write t+1 into slice t;
                                                          // data()[slice_offset(t) + k] == t + 1
TEST(FactorStore, GrowBeyondReservationDetachesAndRegathers)  // cap 1000, grow(5000): detached, values kept;
                                                              // regather(): slices()==1, used()==sum
TEST(FactorStore, ClearKeepsReservationAndTrimReleases)   // grow 64 MiB, clear(), grow 1 MiB, trim():
                                                          // committed_bytes() <= 16 MiB + 2 MiB (test-only accessor)
TEST(FactorStore, MoveAndCopyKeepOffsets)
TEST(FactorStore, ZeroAndTinySlices)                      // reserve_slices(8, 0) and (8, 1); n = 0 grows
TEST(FactorStore, ReservationFailureFallsBackToDetached)  // SMF_TEST_STORE_FAIL_RESERVE=1: every grow detached,
                                                          // regather gives the same values
```

- [ ] **Step 2:** Run `test_factor_store`. Expected: compile failure.
- [ ] **Step 3:** Implement. Test hook `SMF_TEST_STORE_FAIL_RESERVE=1` makes `vm_reserve` return nullptr (read
  once, like `SMF_TEST_STORE_SLICE_CAP` in Task 10).
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "FactorStore"`. Expected: exit 0.
- [ ] **Step 5:** Commit: `feat: FactorStore, per-thread slices of one reserved range`.

### Task 9: `FactorKeep` on `FactorStore`, serial path only

**Files:**
- Modify: `solver/include/smf/factor_posdef.hpp` (`factor_values` and `solve_row_indices` become
  `FactorStore<double>` and `FactorStore<Int>`).
- Modify every reader found by `grep -rn "factor_values\|solve_row_indices" solver/src solver/include`:
  - `factor_indef.cpp`, `factor_posdef.cpp`, `solve*.cpp`, `diagnostics.cpp`, `ipm_session.cpp`, `workspace.cpp`,
    `determinism.hpp`, `smf_ma97_plugin.cpp`, `smf_c.cpp`;
  - `multifrontal_serial.cpp`: the main pass binds slice 0, and `grow(FactorSlice&, ...)` is overloaded.
- Test: the existing suite.

**Interfaces:**
- `FrontPass::fv` becomes `FactorSlice<double> *` and `FrontPass::sri` becomes `FactorSlice<Int> *`.
- `grow(FactorSlice<T> &s, std::size_t need, int64_t &growths)` counts a growth when the slice commits new chunks or
  detaches.
- Entry counts (`Info::actual_factor_entries`, `diagnostics.cpp:6`, `solve.cpp:987`) use `used()`. Bounds checks
  against offsets (`solve_diag.cpp:57,150`) use `size()`.

- [ ] **Step 1:** Make the type change and fix every compile error. The serial path reserves one slice of
  4 x `pred_factor` + `max_f`² entries. The parallel path keeps its per-thread `std::vector` buffers for now and
  gathers into slice 0 of the store via `assign`-like sizing plus `gather_copy`.
- [ ] **Step 2:** Run `rtest.sh $WT/build-mtt` and `rtest.sh $WT/build-seq` (full suites). Expected: both exit 0.
- [ ] **Step 3:** Commit: `refactor: FactorKeep factor and row arrays on FactorStore`.

### Task 10: threads write into their slices, no gather

**Files:** `solver/src/multifrontal_serial.cpp` (`TreeParallelBuffers::Thread`, the `tpass` binding at about line
1333, phase 3 at about lines 1610-1747), `solver/include/smf/info.hpp`, `control.hpp`, `bench_replay.cpp`.
Test: new `solver/tests/test_factor_store_engine.cpp` (`add_test(NAME FactorStoreEngine ...)`).

**Interfaces:**
- `Control` gains `bool factor_in_place = false;`. Environment: `SMF_BENCH_FACTOR_IN_PLACE`.
- `Info` gains `int factor_store_regathers = 0;`.
- With `factor_in_place` on and `threads > 1`:
  - `fkeep.factor_values.reserve_slices(threads, cap)` and the same for `solve_row_indices`. For the factor,
    `cap = 4 * pred_factor + max_f * max_f`; for the row indices, `cap = 4 * pred_rows + max_f`.
  - `tpass[t].fv` is `&fkeep.factor_values.slice(t)` and `tpass[t].sri` is `&fkeep.solve_row_indices.slice(t)`.
  - Phase 3:
    - if both stores are `all_attached()`, then `fbase[t] = slice_offset(t)` and `rbase[t] = slice_offset(t)`, and
      there is no copy;
    - otherwise `regather()` both stores, use today's prefix-sum bases, and `++info.factor_store_regathers`.
  - The relocation and `factor_col_ptr` fix-up loops stay unchanged.
  - After the factorization, `trim()` both stores.
- With it off, the per-thread vectors and `gather_copy` stay as today.

- [ ] **Step 1: Write the failing tests.**

```cpp
TEST(FactorStoreEngine, InPlaceMatchesGather)        // laplacian_2d(200) SPD and opf_like_kkt(3000,1200,1)
    // indefinite at 8 threads, auto_threads off: factor_in_place on vs off -> solutions bitwise equal,
    // same inertia, info.gather_seconds == 0 and factor_store_regathers == 0 when on.
TEST(FactorStoreEngine, OverflowRegathers)           // SMF_TEST_STORE_SLICE_CAP=4096 (test hook read in
    // multifrontal_serial.cpp, caps cap): factor_store_regathers >= 1, solution equal to the gather path.
TEST(FactorStoreEngine, RefactorAcrossThreadCounts)  // one FactorKeep + workspace: 8, 2, 1, 8 threads;
    // after each, solve; x bitwise equal to a fresh FactorKeep factored at the same thread count.
TEST(FactorStoreEngine, TinyAndEmptyMatrices)        // n = 0 (Success, empty store), n = 1, one supernode
    // (5x5 dense KKT) at 8 threads, in place on: solves exact.
TEST(FactorStoreEngine, MovedFactorKeepSolves)       // std::move(fk) into a new FactorKeep, then solve;
    // copy-construct one and solve.
TEST(FactorStoreEngine, ReservationFailureRegathers) // SMF_TEST_STORE_FAIL_RESERVE=1 (Task 8's hook):
    // factor_store_regathers == 1 per factorization, solution bitwise equal to factor_in_place off.
```

- [ ] **Step 2:** Run them. Expected: FAIL (no `factor_in_place`).
- [ ] **Step 3:** Implement. Test hook: `SMF_TEST_STORE_SLICE_CAP`, read once into a function-local static, as
  `test_delay_budget()` reads `SMF_TEST_DELAY_BUDGET` in `smf_ma97_plugin.cpp:194`. Unset means no cap.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "FactorStore|TreeParallel|ParallelDeterminism|RepeatedFactor"`.
  Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: tree-parallel factor written in place in per-thread slices (no gather)`.

### Task 11: Part 2 A/B and PR

- [ ] **Step 1:** Default `factor_in_place = true` and run the 8-case A/B with `SMF_BENCH_FACTOR_IN_PLACE=0` / `1`.
  - Expected: gather 0 (phase split `mf_p3->mf_gathered` under 0.05 ms).
  - Expected: refactor at 8T down by at least 2.0 ms on acopf_grid_160 and pde_control_250.
  - Expected: `peak_rss_bytes` median lower on every case of more than 10 MB of factor.
  - Expected: 1T identical.
- [ ] **Step 2:** Run the board A/B at 8 and 16 threads, `--rounds 1`, metric wall. Pass: each track's geomean at
  most 1.005. Run the full `rtest.sh` on both builds.
- [ ] **Step 3:** Push and open the PR (against `perf/8t-p4-valueprep`). Update `STATUS.md`.

### Part 3: solve preparation off the serial path (branch `perf/8t-p3-layout`)

### Task 12: measure the layout sub-steps

- [ ] **Step 1:** Run `rpy.sh e8t_p3_layout "SMF_BENCH_PHASE_TRACE=1" phase_run.py mtt -- --threads 8` on the Part 2
  tip, then `phase_split.py`.
- [ ] **Step 2:** Write the table of `mf_gathered->mf_steps`, `mf_steps->mf_diag`, `mf_diag->mf_top`,
  `mf_top->mf_toprows` and `mf_toprows->mf_ret` per case into the Part 3 PR body draft.
  - A sub-step under 0.1 ms on all 8 cases is dropped: skip its task below and say so in the PR.
  - `mf_toprows->mf_ret` is `finish_solve_layout` (Task 13).
  - `mf_diag` is Task 15.
  - `mf_top` and `mf_toprows` are Task 14.

### Task 13: D^-1 per front

**Files:** `solver/src/solve_diag.cpp` (split out the per-front kernel), `solver/include/smf/factor_posdef.hpp`
(`dinv` and `dinv_kind` become `FactorStore<double>` and `FactorStore<std::int8_t>`), `multifrontal_serial.cpp`
(`process_front` after the factor is stored, and the step loop). Test: `solver/tests/test_solve_diag.cpp`.

**Interfaces:**
- Produces `void front_inline_dinv(const double *L, int p, std::ptrdiff_t f, double *d, std::int8_t *kind) noexcept;`.
  This is the body of `build_inline_dinv`'s per-step loop, unchanged in arithmetic, and `build_inline_dinv` calls it.
- With `Control::per_front_dinv` on (environment `SMF_BENCH_PER_FRONT_DINV`), on the tree-parallel path, when
  `small_solve_kernels && inline_d && !posdef`:
  - each front appends 3p entries to `fkeep.dinv.slice(t)` and p entries to `fkeep.dinv_kind.slice(t)`;
  - it records the absolute offset in a per-supernode array, and the step loop writes `fkeep.dinv_offset[k]` from
    it;
  - `finish_solve_layout` skips `build_inline_dinv` when the engine already built `dinv` (a flag in `FactorKeep`).
- Slice capacity: `4 * n` pivots per slice. Overflow detaches and regathers, as in Task 10.

- [ ] **Step 1: Write the failing test.** `TEST(SolveDiag, PerFrontDinvEqualsSerialBuild)`:
  - use `opf_like_kkt(3000, 1200, 1)` and `cascade_kkt(40, 600, 4.1, 3.0)` (2x2 pivots and delays) at 8 threads;
  - for every step k, the 3p `dinv` entries at `dinv_offset[k]` and the kind entries are bitwise equal to a
    factorization with `per_front_dinv` off;
  - the solutions are bitwise equal.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "SolveDiag|SolveInterleaved|TreeParallel|SolveEndToEnd"`.
  Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: D^-1 of the solve built per front in the parallel phases`.

### Task 14: fixed top order and the solve-layout cache

**Files:** `multifrontal_serial.cpp` (the non-`det` top block at about lines 1711-1742, `solve_top_rows` at about
1765-1808, `TreeParallelBuffers`), `control.hpp`, `bench_replay.cpp`. Test: `solver/tests/test_tree_parallel.cpp`.

**Interfaces:**
- `Control::cached_solve_layout`, environment `SMF_BENCH_CACHED_SOLVE_LAYOUT`.
- `TreeParallelBuffers` gains `struct SolveLayoutCache` with these members:
  - `std::uint64_t analysis = 0;`
  - `int threads = 0;`
  - `std::vector<Int> top_parent;`
  - `std::vector<Int> top_rows;`
  - `std::vector<Int> top_row_lists;` (the top steps' rows, concatenated)
  - `std::vector<int> top_rows_count;`
  - `int solve_threads = 1;`
- With the flag on, the top steps are appended in the ascending postorder position of their nodes, not in
  `done_order`.
  - `solve_top_parent` comes from the cache when `analysis` and `threads` match.
  - `solve_top_rows`, `solve_top_slot` and `solve_threads` are reused when the top steps' realized row lists equal
    `top_row_lists` (one `std::equal`). Otherwise they are rebuilt as today, and the cache is refilled.
  - Any error return clears the cache (`analysis = 0`).

- [ ] **Step 1: Write the failing tests.**

```cpp
TEST(TreeParallel, LayoutCacheReusedOnUnchangedSequence)  // laplacian_2d(200), 8 threads, 3 refactors:
    // fk.solve_top_parent identical across refactors and solve residual < 1e-12; Info::solve_layout_reused
    // (new int) == 0, 1, 1.
TEST(TreeParallel, LayoutCacheRebuiltWhenDelaysMoveRows)  // cascade_kkt V0 then V1 values: reused == 0 on V1,
    // solution residual < 1e-10 (refined as test_factor_values_reuse does), then V1 again: reused == 1.
TEST(TreeParallel, LayoutCacheClearedAfterAbort)          // InertiaExceeded on V1, then V0: reused == 0, solve ok.
```

- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement as described, with the new field `Info::solve_layout_reused`.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "TreeParallel|ParallelDeterminism|Solve"`. Expected: exit 0.
- [ ] **Step 5:** Commit: `perf: fixed top solve order; solve layout cached across refactorizations`.

### Task 15: `diag_entries` on demand and the 2x2 counter

**Files:** `solver/include/smf/factor_posdef.hpp`, `multifrontal_serial.cpp` (the `diag_entries` block at about
lines 1659-1685), `solver/src/pivot_order.cpp`, `solver/src/solve.cpp:707`, `solver/src/solve_diag.cpp:214`,
`solver/src/smf_ma97_plugin.cpp` (lines 733, 797, 951), `info.hpp`. Test: `solver/tests/test_solve_diag.cpp`,
`solver/tests/test_ipopt_plugin_runtime.cpp`.

**Interfaces:**
- `const std::vector<DiagSolveEntry> &FactorKeep::diag() const;` builds `diag_entries` on first call from the factor
  (the same order: fronts by postorder position, pivots in order) when the engine left it unbuilt. Every reader
  calls `diag()`.
- `Info::two_by_two_pivots` (int) is counted per front in the kernels' stats. The plugin reads it instead of
  scanning.
- `Control::lazy_diag_entries`, environment `SMF_BENCH_LAZY_DIAG`.

- [ ] **Step 1: Write the failing tests.**
  - `TEST(SolveDiag, LazyDiagEqualsEager)`: on `cascade_kkt` at 8 threads, `fk.diag()` equals the eager
    `diag_entries`, element by element, with the flag off.
  - `TEST(SolveDiag, TwoByTwoCounterMatchesScan)`: `info.two_by_two_pivots` equals the count of `tag == 2`.
  - The plugin test checks `num_two` is unchanged.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `rtest.sh $WT/build-mtt "SolveDiag|IpoptPlugin|Ma97Plugin|PivotOrder|CApi"`. Expected:
  exit 0.
- [ ] **Step 5:** Commit: `perf: diag_entries built on demand; 2x2 count from the factorization stats`.

### Task 16: Part 3 A/B, determinism and PR

- [ ] **Step 1: Write the failing test.**

```cpp
TEST(ParallelDeterminism, Engine8tStage1PartsOn) {
    // Deterministic mode, all Stage 1 flags on (front costs, parallel value prep, in place, per-front dinv,
    // cached layout, lazy diag): opf_like_kkt and laplacian; 1 vs 8 threads, 3 refactors each:
    // solutions bitwise equal, inertia equal.
}
```

  Run it; if it already passes, keep it as a guard.
- [ ] **Step 2:** Default the Part 3 flags on. Run the 8-case A/B with all three flags off / on, and the phase split.
  - Expected: `mf_gathered->mf_ret` under 0.5 ms on every case.
  - Expected: no case slower.
- [ ] **Step 3:** Run the full `rtest.sh` on both builds. Expected: exit 0.
- [ ] **Step 4:** Push and open the PR (against `perf/8t-p2-store`). Update `STATUS.md`.

### Task 17: Stage 1 gate

Run this on the tip of `perf/8t-p3-layout`, or on `main` if Daniel has merged the stack.

- [ ] **Step 1:** Re-measure the rivals and the stage tip on the same day:

  `rrun.sh e8t <bench_replay> e8t_gate1 - -- --manifest benchmarks/manifests/probe_refactor8t.json --threads 1,8
  --trials 3 --warmups 1 --allow-short-trials --timeout 300 --seq-steps all --solvers smf,mkl_pardiso`

  Then `project_run.py --smf e8t_gate1 --rivals e8t_gate1 --threads 8 --gate stage1`. Pass: exit 0. Record the
  acopf ratios.
- [ ] **Step 2: Board.** Run `rrun.sh ... e8t_board_gate1 ... --threads 1,8,16 --solvers smf` (as Task 0 Step 11),
  then `ab_compare.py --a e8t_board_base --b e8t_board_gate1 --metric wall`. Pass:
  - each track's geomean at most 1.005;
  - no case at 1.05 or worse on a 3-round rerun;
  - `score.py --inertia-ref benchmarks/manifests/fronttier_inertia.json` shows 0 wrong;
  - validation outcomes identical;
  - no backward error more than 2x worse on a case passing at base.
- [ ] **Step 3: Peak RSS.** Pass: the median per case in `e8t_gate1` is at most the one in `e8t_base`, at 8T.
- [ ] **Step 4: IPOPT end-to-end** on the 8 cases with the plugin (the PR #64 r2 configuration,
  `--track-override ipopt-plugin`). Pass:
  - same termination;
  - identical iteration counts, or each difference traced to rounding (the same run at 1T reproduces it) before
    calling it a pass.
- [ ] **Step 5:** Write `docs/performance/2026-11-XX-8t-engine-stage1-gate.md` (the date of the run): the tables,
  the verdict, and the acopf ratios against the Stage 2 targets (about 0.91 / 0.94).
- [ ] **Step 6: Decide.**
  - **Pass:** report to Daniel and write the Stage 2 plan (Parts 5-7) with the writing-plans skill.
  - **Fail on any of the six cases:** stop, report the numbers, and re-plan with Daniel (spec stop rule).
  - Either way, update `STATUS.md` and the memory.
