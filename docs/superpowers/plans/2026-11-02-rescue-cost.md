# Rescue Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the scale-on-delay rescue cheaper on delay-heavy small factors (ncvxqp9 rec 32 ms vs MA97 16 ms, stokes128 rec, PGLib rec) in two ways: it resumes the matcher the analysis already built, and it cheapens or skips the stopped unscaled first try after an initial rejection.

**Architecture:**
- On an initial rejection (`MatchingTrial::RejectedInitial`), the analysis now throws away a `SymmetricMatcher` that already holds the cost graph, the greedy and length-2 phase, and the budgeted searches. The analysis will keep that matcher, as it stands at the budget point, in the `AnalysisKeep`.
  - The concurrent trial forks it at the budget point, so the kept state is the same at every thread count.
- When the rescue (`matching_scale_of` in `solve.cpp`) factors the analysed values, it forks the kept matcher. It then runs the remaining searches up to the 2%-free stop and scales by those duals. It no longer rebuilds the cost graph and redoes the searches.
- A new `Control::scale_on_delay_first_ratio` caps the first try's work limit, or skips the try, on keeps whose trial rejected on the initial matching. A board picks its default.

**Tech Stack:** C++20, MSVC 2022, CMake, GoogleTest, Python 3.12 bench tools (`tools/bench/ab_run.py`, `ab_compare.py`, `score.py`).

**Spec:** the recommendation at `https://claude.ai/artifact/Dvh52kmg9CwrigE8smPqUL` (plan 2), plus these write-ups:
- `docs/performance/2026-11-02-factor-side-delays.md` (section 3 and "Open");
- `docs/performance/2026-10-31-analyse-gap.md` ("A nearly complete initial matching keeps its scale", "Where the analysis time went", "Open levers");
- `docs/performance/2026-11-01-matching-search-order.md` ("Measuring").

## Background (numbers this plan argues from)

ncvxqp9 rec, 1 thread, after PR #18: wall 32.0 ms, factor 23.0 ms. MA97 takes 16.0 ms wall and 10.3 ms factor, without scaling.

| piece of factor() | ms |
|---|---:|
| first try, unscaled, stopped by the work limit (16x delay-free work; it stops at 19x) | 5.5 |
| rescue matching (`SymmetricMatcher(C)`, `set_stop_free(2% n)`, Column order) | 11.6 |
| rescaled factor | 4.5 |

- About 8 ms of the 11.6 ms are the matcher's constructor (`build_cost_graph`, initial duals, greedy, length-2). The analysis ran these on the same values, in Degree order, plus its budgeted searches (19.5% free at the budget point, 31% after the greedy phase), and then threw them away (`paused_matcher.reset()` / `concurrent->stop()` in `symbolic_analysis.cpp`).
- stokes128 rec is an initial rejection with 1.0% free. Its partial scale is kept (`kInitialScaleMaxFree`). If its first try stops, a fresh rescue matcher stops at its first search: 1% free is already at most 2%. The rescue then pays a whole constructor for duals close to the ones already applied.
- PGLib rec rescues paid 67-370 ms per full matching before the 2% stop. They are initial rejections too (13.5-15% free).
- Doing nothing about the matching does not work. Letting ncvxqp9's unscaled factor finish takes 19 ms with the new kernel, and MC64 scaling in MA97 costs about 25 ms on its own.

Projected effect on ncvxqp9 rec: factor 23 → about 8-12 ms, wall about 18-22 ms. These are estimates. The gates below decide.

## Global Constraints

- Windows, MSVC 2022, CMake. cmake and ctest are not on PATH. Their directory is `"/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/"` (`$CMAKE`, `$CTEST` below).
- Build directory: the worktree's `build/`. Build with `--config Release --parallel 4` under `nice -n 10`. The user games on this machine.
- Tests that use CoinHSL need `SMF_HSL_LIBRARY=C:/Users/daniel/projects/opt_solver/build-deps/coinhsl-2024.5.15/CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5/bin/libhsl.dll`, with that `bin` directory on PATH.
- Threads are `std::thread` only, never OpenMP.
- Deterministic mode (`Control::deterministic`) keeps bitwise-identical factors, inertia, scales and solutions at every `num_threads`. `MatchingTrial.ConcurrentTrialIsTheSerialOne`, `ScaleOnDelay.DeterministicAcrossThreadCounts` and `ParallelDeterminism` must pass.
- The trial's verdict must not change. `info.matching_trial`, `ordering_used`, `perm`, `matching_scale` and `rejected_matching_scale` stay bitwise equal to main on every `MatchingTrial.*` test.
- Accuracy gate (score.py): eta ≤ 1e-10, omega ≤ 1e-8. No case-track may lose accuracy.
- Benchmarks:
  - A and B come from one binary on one machine, interleaved (`tools/bench/ab_run.py`), with the variant chosen by environment variable;
  - front-tier manifest `benchmarks/manifests/fronttier.json`, 1 thread, 3 rounds unless stated;
  - run one at a time, check first that no `bench_replay.exe` is running, run under below-normal priority, and announce any multi-core run longer than 10 minutes;
  - A/A noise floor of this setup: ±5% per case, ±0.3% per class (factor-side-delays write-up).
- Compare matcher work by edges scanned (`SymmetricMatcher::work()`), not standalone matcher ms (±20% drift on the Ryzen).
- Never push. Never commit CoinHSL files, the `.gitignore` change, or `*.log` files.
- Code style: 2-space indent in `solver/src/`, 4 in `solver/include/` and `solver/tests/`, matching each file. Every new `Control` or `Info` field gets a comment with the evidence for its default.

## Review Focus

1. **Values other than the analysed ones** (every IPOPT step, an `updated` matrix in the rec track). A kept matcher describes the analysed values only. Rescuing other values with it would scale by the wrong duals. Tested in Task 4: `ScaleOnDelay.OtherValuesGetAFreshMatching`.
2. **Thread count at analysis.** The concurrent trial's worker keeps searching past the budget until it is stopped. Keeping that matcher as it is would make the rescue scale depend on timing. Tested in Task 3: the extended `MatchingTrial.ConcurrentTrialIsTheSerialOne` compares the kept matcher's `current().scale` and `work()` at 1 and 4 threads. Task 4's `ScaleOnDelay.ReusedMatcherDeterministicAcrossThreads` compares the rescued factor.
3. **Memory held past the analysis.** The cost graph is O(nnz): PGLib 9241 has n = 400k. It must be released after the first armed factor of the keep, on every return path of `factor_scale_on_delay`, and never kept when the rescue cannot fire. Tested in Task 3 (`MatchingTrial.KeptMatcherOnlyWhenTheRescueIsArmed`) and Task 4 (`ScaleOnDelay.RescueResumesTheAnalysisMatcher` asserts `ak->rescue_matcher == nullptr` afterwards). The board in Task 4 checks `rss B/A` per case.
4. **Skipping the first try where the unscaled factor would have passed.** An initial rejection whose unscaled factor stays under 16x is factored scaled for nothing, and may be less accurate (the pglib_opf_case4661_sdet history). Task 5's board decides on the tuning split, with a per-case loss cap, and confirms on heldout. Tested in Task 5: `ScaleOnDelay.FirstRatioIgnoredWithoutInitialRejection` pins that keeps which are not initial rejections are untouched.
5. **A kept matcher that later turns singular.** A search can fail in phase 3, or a column can stay unmatched. The rescue must then fall back exactly as an empty scale does today: run unarmed and record the zero signature. Tested in Task 4: `ScaleOnDelay.KeptMatcherSingularFallsBack`, which uses a pattern with an all-zero column whose initial matching rejects.

---

### Task 1: Rescue piece timers and the baseline

**Files:**
- Modify: `solver/include/smf/info.hpp` (next to `delay_stops`), `solver/src/solve.cpp` (`factor_scale_on_delay`)
- Modify: `solver/benchmarks/adapters/adapter.hpp` (AdapterStats), `solver/benchmarks/adapters/hsl.cpp:339` (fill them), `solver/benchmarks/common/result.hpp` + `solver/benchmarks/common/validation.cpp:175` (record fields), `solver/benchmarks/bench_replay.cpp:758` (`--print-steps` line) and `:891` (cold record)
- Test: `solver/tests/test_scale_on_delay.cpp`

**Interfaces:**
- Produces:
  - `Info::delay_try_seconds` (double): time of a stopped or failed try, the one now folded into `extra_seconds`. 0 when no try was spent.
  - `Info::delay_matching_seconds` (double): time inside `matching_scale_of`. 0 when no matching ran.
  - Both are reset with `delay_rescales` in the three `Solver::factor` overloads, and both are already included in `factor_seconds`.
  - `AdapterStats::delay_try_ms, delay_matching_ms`; records `"delay_try_ms"`, `"delay_matching_ms"` (written when ≥ 0, default -1 as `delay_rescales`); `--print-steps` appends ` try_ms=%.3f match_ms=%.3f`.

- [ ] **Step 1: Write the failing test** `ScaleOnDelay.RescuePiecesTimed`:

```cpp
const CscLower A = kkt(kG, kM, 4.1, 3.0);          // fires (see FiresOnBadlyScaledKkt)
// factor once into a fresh FactorKeep
EXPECT_EQ(i1.delay_stops, 1);
EXPECT_GT(i1.delay_try_seconds, 0.0);
EXPECT_GT(i1.delay_matching_seconds, 0.0);
EXPECT_LE(i1.delay_try_seconds + i1.delay_matching_seconds, i1.factor_seconds);
// second factor, same values, same FactorKeep: kept scale reused
EXPECT_EQ(i1b.delay_try_seconds, 0.0);
EXPECT_EQ(i1b.delay_matching_seconds, 0.0);
// well scaled kkt(kG, kM, 4.1, 0.0): both 0
```

- [ ] **Step 2: Run it and check it fails**

Run: `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 --target test_scale_on_delay`
Expected: compile error, no member `delay_try_seconds`.

- [ ] **Step 3: Implement.** In `factor_scale_on_delay`, the two `extra_seconds += info.factor_seconds` sites also add to `info.delay_try_seconds`, and the `matching_scale_of` timing (`t0` before the call) goes to `info.delay_matching_seconds`. Then pass both through to the adapter, the record and the print line.

- [ ] **Step 4: Run the tests and check they pass**

Run: `"$CTEST" --test-dir build -C Release -R "ScaleOnDelay|MatchingTrial" --output-on-failure`
Expected: `100% tests passed`.

- [ ] **Step 5: Record the baseline** (1T, rec track, 3 rounds, smf only). Announce first: about 10 minutes.

Run: `python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json --cases ncvxqp1,ncvxqp9,stokes128,cops_elec_200,pglib_opf_case300_ieee,pglib_opf_case500_goc__api,pglib_opf_case1354_pegase,pglib_opf_case2869_pegase,pglib_opf_case4661_sdet,darcy003,mario002 --rounds 3 --output benchmarks/results/rescue_cost_base --binary build/benchmarks/Release/bench_replay.exe --variant base=SMF_BENCH_SCALE_ON_DELAY=16`

For each recommended-track case, write down `analyse_ms`, `factor_ms`, `delay_try_ms`, `delay_matching_ms`, `delay_rescales`, delays and the analysis' `matching_trial` (from `bench_matching_trial --list` on the same cases). Put them in a table in the commit message. Expected: ncvxqp9 try about 5.5 ms and matching about 11.6 ms, with the trial RejectedInitial. Also record whether stokes128 rec stops its first try.

- [ ] **Step 6: Commit**

```bash
git add solver/include/smf/info.hpp solver/src/solve.cpp solver/benchmarks solver/tests/test_scale_on_delay.cpp
git commit -m "feat(rescue): time the stopped try and the rescue matching"
```

### Task 2: `SymmetricMatcher::fork()`

**Files:**
- Modify: `solver/include/smf/scaling_matching.hpp:40-80`, `solver/src/scaling_matching.cpp:200-390` (State, constructor, `result`)
- Test: `solver/tests/test_refinement.cpp` (the `Matching.*` group, next to `Matching.StopFreeLeavesAPartialMatching`)

**Interfaces:**
- Produces: `std::unique_ptr<SymmetricMatcher> SymmetricMatcher::fork() const;`
  - The fork shares the immutable cost graph: `State::g` becomes `std::shared_ptr<const CostGraph>`, and every `s.g.` read becomes `s.g->`.
  - It copies the matching and dual state: `rs`, `row_match`, `row_dense`, `v`, `col_to_row`, `matched`, `any_dense`, `phase`, `next_col`, `order`, `search_order`, `work`, `status`, `stop_free`, and the `init_*` arrays.
  - Its search scratch is sized as the constructor sizes it: `heap.reset(n)`, `touched`/`finalized`/`level` sized n, `row_parent` -1, `row_done` 0.
  - Cost O(n), not O(nnz).
  - A private constructor `SymmetricMatcher(std::unique_ptr<State>)` is the fork's way in.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST(Matching, ForkResumesLikeTheOriginal) {
    const CscLower A = opf_like_kkt(6000, 5000, 7);     // kkt_test_matrices.hpp
    SymmetricMatcher m(A, MatchingSearchOrder::Degree);
    ASSERT_FALSE(m.advance(2 * static_cast<long long>(A.row_idx.size())));  // paused mid-way
    auto f = m.fork();
    EXPECT_EQ(f->work(), m.work());
    m.advance(); f->advance();
    const MatchingResult a = m.current(), b = f->current();
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.col_to_row, b.col_to_row);
    EXPECT_EQ(a.scale, b.scale);                         // bitwise
    EXPECT_EQ(f->work(), m.work());
}
TEST(Matching, ForkLeavesTheOriginalUntouched) {
    // pause m as above; record r0 = m.current(), w0 = m.work()
    // f = m.fork(); f->set_stop_free(0); f->advance();
    EXPECT_EQ(m.work(), w0);
    EXPECT_EQ(m.current().col_to_row, r0.col_to_row);
    EXPECT_EQ(m.current().scale, r0.scale);
    // and m.advance() afterwards equals a never-forked matcher's result
}
```

- [ ] **Step 2: Run them and check they fail**

Run: build `test_refinement`. Expected: compile error, no member `fork`.

- [ ] **Step 3: Implement `fork()`** as specified under Interfaces. `build_cost_graph` fills a fresh `CostGraph` that is then moved into the `shared_ptr`.

- [ ] **Step 4: Run the tests and check they pass**

Run: `"$CTEST" --test-dir build -C Release -R "Matching|MatchingTrial|ScaleOnDelay" --output-on-failure`
Expected: `100% tests passed`. Results do not change: `fork` is not used yet.

- [ ] **Step 5: Commit**

```bash
git add solver/include/smf/scaling_matching.hpp solver/src/scaling_matching.cpp solver/tests/test_refinement.cpp
git commit -m "feat(matching): SymmetricMatcher::fork shares the cost graph"
```

### Task 3: The analysis keeps its matcher at the budget point

**Files:**
- Modify: `solver/include/smf/analysis.hpp:55-85` (AnalysisKeep), `solver/include/smf/control.hpp` (after `scale_on_delay_reuse_tol`)
- Modify: `solver/src/symbolic_analysis.cpp`: `ConcurrentTrial::run` (fork after `matcher.advance(budget_)` returns false, before `phase1_ready_` is published), the serial `paused_matcher` branch (~line 765), and the keep assembly (~line 970)
- Modify: `solver/benchmarks/bench_replay.cpp:557` (env `SMF_BENCH_RESCUE_KEEP_MATCHER`)
- Test: `solver/tests/test_matching_trial.cpp`

**Interfaces:**
- Consumes: `SymmetricMatcher::fork()` (Task 2).
- Produces:
  - `AnalysisKeep::matching_trial` (`MatchingTrial`, default `None`): the analysis' `info.matching_trial`.
  - `mutable std::shared_ptr<const SymmetricMatcher> AnalysisKeep::rescue_matcher`: on `RejectedInitial` only, the trial's matcher at the budget point (`advance(budget)` returned false, nothing after that).
    - It is kept only when the analysis Control could arm the rescue: `scale_on_delay_keep_matcher`, `scale_on_delay_ratio >= 0`, `scaling == ScalingMethod::Auto`, `pivot_u >= kWeakPivotRatio`.
    - It is `mutable` because `factor()` releases it. The comment repeats the keep's contract: one keep is not factored from two threads at once.
  - `bool Control::scale_on_delay_keep_matcher = true;`, with a comment carrying the Task 4 evidence. False gives main's behaviour (A/B switch).
  - `ConcurrentTrial::take_budget_fork()` returns the worker's fork (`std::shared_ptr<const SymmetricMatcher>`) after `finish()`.

- [ ] **Step 1: Write the failing tests:**

```cpp
TEST(MatchingTrial, RejectedInitialKeepsTheBudgetMatcher) {
    // As RejectsOnInitialMatchingPastBudget (opf_like_kkt(6000, 5000, 7)), with
    // matching_search_budget = 0.5 so that some searches ran (0.0 runs none).
    ASSERT_EQ(info.matching_trial, MatchingTrial::RejectedInitial);
    EXPECT_EQ(ak->matching_trial, MatchingTrial::RejectedInitial);
    ASSERT_NE(ak->rescue_matcher, nullptr);
    SymmetricMatcher ref(ak->cleaned, MatchingSearchOrder::Degree);
    const long long budget = static_cast<long long>(0.5 * 2.0 * ak->cleaned.row_idx.size());
    ASSERT_FALSE(ref.advance(budget));
    EXPECT_EQ(ak->rescue_matcher->work(), ref.work());
    EXPECT_EQ(ak->rescue_matcher->current().scale, ref.current().scale);
}
TEST(MatchingTrial, KeptMatcherOnlyWhenTheRescueIsArmed) {
    // same matrix and budget; each line is its own analysis:
    // pivot_u = 1e-8                      -> rescue_matcher == nullptr
    // scale_on_delay_ratio = -1           -> nullptr
    // scaling = ScalingMethod::None       -> nullptr
    // scale_on_delay_keep_matcher = false -> nullptr, and perm/matching_scale equal the default run's
    // budget -1 (Exact)                   -> nullptr, matching_trial Exact
    // diagonal_kkt(...) (Diagonal)        -> nullptr
}
```

Extend `MatchingTrial.ConcurrentTrialIsTheSerialOne`: when `a1->rescue_matcher` is set, `a4->rescue_matcher` is set too, with equal `work()` and bitwise equal `current().scale` and `current().col_to_row`. When it is not set, `a4->rescue_matcher` is null.

- [ ] **Step 2: Run them and check they fail**

Run: build `test_matching_trial`. Expected: compile error, no member `rescue_matcher`.

- [ ] **Step 3: Implement**
  - Serial path: on rejection (`initial_passes` false), move `paused_matcher` into a local `std::shared_ptr<const SymmetricMatcher>` instead of resetting it.
  - Concurrent path: the worker forks before it publishes `phase1_ready_`. The calling thread takes the fork after `finish()` on the stop branch.
  - Keep assembly: set `keep->matching_trial`. Set `keep->rescue_matcher` only under the armed condition and when `info.matching_trial == RejectedInitial`.
  - bench_replay: `SMF_BENCH_RESCUE_KEEP_MATCHER=0|1` → `ctrl.scale_on_delay_keep_matcher`.

- [ ] **Step 4: Run the tests and check they pass**

Run: `"$CTEST" --test-dir build -C Release -R "MatchingTrial|ScaleOnDelay|ParallelDeterminism" --output-on-failure`
Expected: `100% tests passed`. Factors do not change yet (nothing reads `rescue_matcher`).

- [ ] **Step 5: Commit**

```bash
git add solver/include/smf/analysis.hpp solver/include/smf/control.hpp solver/src/symbolic_analysis.cpp solver/benchmarks/bench_replay.cpp solver/tests/test_matching_trial.cpp
git commit -m "feat(analysis): keep the rejected trial's matcher at the budget point"
```

### Task 4: The rescue resumes the kept matcher

**Files:**
- Modify: `solver/src/solve.cpp:110-145` (`matching_scale_of`), `:265-420` (`factor_scale_on_delay`: release on every return), `solver/include/smf/info.hpp`
- Modify: `solver/tests/kkt_test_matrices.hpp` (move `scaled_opf_kkt` here from `test_matching_trial.cpp`)
- Test: `solver/tests/test_scale_on_delay.cpp` (and update its `matching_scale_of` helper to mirror the new rule)

**Interfaces:**
- Consumes: `AnalysisKeep::rescue_matcher`, `SymmetricMatcher::fork()`.
- Produces:
  - `matching_scale_of(ak, cv)` checks first for an analysis' own full scale (unchanged). Otherwise, when `ak.rescue_matcher` is set and `cv == ak.cleaned.values`, it forks the kept matcher, calls `set_stop_free(kRescueMaxFree * n)` and `advance()`, and takes `current()`.
    - It applies the same scale checks as today: status, size, positive and finite.
    - Other values: a fresh `SymmetricMatcher(C)`, Column order, unchanged.
  - `factor_scale_on_delay` resets `ak.rescue_matcher` before it returns, on every path. It does so before the rescaled `run(...)`, so the graph is not held through the second factor.
  - `Info::delay_matcher_reused` (int): 1 when this factor's rescue matching resumed the analysis' matcher, else 0. It is reset with `delay_rescales`.

- [ ] **Step 1: Find the fixture and write the failing tests.** Fixture: `scaled_opf_kkt(6000, 5000, 7, 3.0)`, analysed with `base_control()` and `matching_search_budget = 0.5`.
  - Premise assertions: `info.matching_trial == RejectedInitial` and `unscaled_ratio(A, ctrl) > ctrl.scale_on_delay_ratio`.
  - If the premise fails, raise `spread` in steps of 1.0 up to 6.0. Never change the assertions after the premise.

```cpp
TEST(ScaleOnDelay, RescueResumesTheAnalysisMatcher) {
    ASSERT_NE(ak->rescue_matcher, nullptr);
    // reference: a Degree matcher paused at the analysis' budget, then stopped at 2% free
    SymmetricMatcher ref(ak->cleaned, MatchingSearchOrder::Degree);
    ref.advance(budget_edges);  ref.set_stop_free(int(0.02 * n));  ref.advance();
    const std::vector<double> expect = ref.current().scale;
    // factor the analysed values
    EXPECT_EQ(i1.delay_rescales, 1);
    EXPECT_EQ(i1.delay_matcher_reused, 1);
    EXPECT_EQ(fk.scale, expect);                 // bitwise
    EXPECT_EQ(ak->rescue_matcher, nullptr);      // released
    expect_accurate_solve(s, fk, ctrl, A);
}
TEST(ScaleOnDelay, OtherValuesGetAFreshMatching) {
    // analyse A, factor B = A with H diagonal x1.01 (same pattern) first
    EXPECT_EQ(i.delay_matcher_reused, 0);
    EXPECT_EQ(fk.delay_scale, matching_scale_of(*ak, B));   // the Column-order fresh one
}
TEST(ScaleOnDelay, KeptMatcherReleasedWhenTheRescueDoesNotFire) {
    // a RejectedInitial fixture whose unscaled ratio stays under the limit
    // (scaled_opf_kkt(6000, 5000, 7, 0.0)); premise ASSERT on both
    EXPECT_EQ(i.delay_stops, 0);
    EXPECT_EQ(ak->rescue_matcher, nullptr);
}
TEST(ScaleOnDelay, KeptMatcherSingularFallsBack) {
    // the fixture with one column's values all zeroed (pattern kept), premise RejectedInitial
    // -> factor ran (Singular or Success), delay_rescales == 0, fk.delay_signature set, and the
    //    next factor of the same values runs no try (delay_stops == 0)
}
TEST(ScaleOnDelay, ReusedMatcherDeterministicAcrossThreads) {
    // deterministic = true, analyse+factor at num_threads 1 and 4:
    // equal fk.scale, delayed_pivots, inertia, and the solution bitwise
}
```

- [ ] **Step 2: Run them and check they fail**

Run: `"$CTEST" --test-dir build -C Release -R ScaleOnDelay --output-on-failure`
Expected: `RescueResumesTheAnalysisMatcher` fails on `delay_matcher_reused` / `fk.scale`, and the release assertions fail.

- [ ] **Step 3: Implement** `matching_scale_of` and the release, as specified under Interfaces.

- [ ] **Step 4: Run the tests and check they pass**

Run: `"$CTEST" --test-dir build -C Release -R "ScaleOnDelay|MatchingTrial|Matching|ParallelDeterminism" --output-on-failure`
Expected: `100% tests passed`.

- [ ] **Step 5: Edges first.** For the rec-track RejectedInitial cases from Task 1's table, log `SymmetricMatcher::work()` of the rescue matching, kept and fresh, with a temporary `fprintf` under `SMF_BENCH_RESCUE_TRACE`. Remove it before you commit.
  - Expected: the kept rescue scans fewer edges on every case.
  - On ncvxqp9 it scans at most the fresh rescue's edges minus the analysis' budgeted work.

- [ ] **Step 6: A/B gate.** Indefinite front tier, 1T, 3 rounds. Announce first: about 40 minutes.

Run: `python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json --classes small_kkt,large_kkt,hard_indef --rounds 3 --output benchmarks/results/rescue_keep_ab --binary build/benchmarks/Release/bench_replay.exe --variant off=SMF_BENCH_RESCUE_KEEP_MATCHER=0 --variant keep=SMF_BENCH_RESCUE_KEEP_MATCHER=1`
then `python tools/bench/ab_compare.py --a benchmarks/results/rescue_keep_ab/off --b benchmarks/results/rescue_keep_ab/keep --manifest benchmarks/manifests/fronttier.json --metric wall` (tuning and heldout separately).

Pass:
- no case-track loses accuracy;
- recommended-track geomean ≤ 1.00 on both splits;
- no case > 1.05x wall outside the A/A floor;
- ncvxqp1 rec and ncvxqp9 rec factor B/A < 0.85;
- `rss B/A` ≤ 1.10 on every case.

If any of these fails, stop and report the failing rows. Do not tune.

- [ ] **Step 7: Commit** (put the gate table in the message and update the `Control::scale_on_delay_keep_matcher` comment with it)

```bash
git add solver/src/solve.cpp solver/include/smf/info.hpp solver/include/smf/control.hpp solver/tests/kkt_test_matrices.hpp solver/tests/test_matching_trial.cpp solver/tests/test_scale_on_delay.cpp
git commit -m "perf(rescue): resume the analysis matcher on the analysed values"
```

### Task 5: Cap or skip the first try after an initial rejection

**Files:**
- Modify: `solver/include/smf/control.hpp` (after `scale_on_delay_next_ratio`), `solver/src/solve.cpp` (`factor_scale_on_delay`, the `first_try` branch at `try_limit`)
- Modify: `solver/benchmarks/bench_replay.cpp:559` (env `SMF_BENCH_SCALE_ON_DELAY_FIRST`)
- Test: `solver/tests/test_scale_on_delay.cpp`

**Interfaces:**
- Consumes: `AnalysisKeep::matching_trial` (Task 3), `Info::delay_try_seconds` (Task 1).
- Produces: `double Control::scale_on_delay_first_ratio = -1.0;`, with its default set by Step 6.
  - It applies only to the very first armed factor of a keep into a FactorKeep (`!ours`), when `ak.matching_trial == MatchingTrial::RejectedInitial`.
  - Value > 0: `try_limit = min(limit, floor(first_ratio * free_work))`.
  - Value 0: no try. Control goes straight to the "new matching scaling" block, with `delay_stops = 0`.
  - Negative: no change.

- [ ] **Step 1: Write the failing tests** (fixture as in Task 4):

```cpp
TEST(ScaleOnDelay, SkippedFirstTryGivesTheSameRescue) {
    // default run (first_ratio -1) vs first_ratio 0, fresh analysis + FactorKeep each
    EXPECT_EQ(i_def.delay_stops, 1);
    EXPECT_EQ(i_skip.delay_stops, 0);
    EXPECT_EQ(i_skip.delay_rescales, 1);
    EXPECT_EQ(i_skip.delay_try_seconds, 0.0);
    EXPECT_EQ(fk_skip.scale, fk_def.scale);              // the scale does not depend on the try
    EXPECT_EQ(i_skip.delayed_pivots, i_def.delayed_pivots);
    EXPECT_EQ(i_skip.num_negative, i_def.num_negative);
}
TEST(ScaleOnDelay, FirstRatioCapsTheTry) {
    // first_ratio 4: delay_stops 1, same fk.scale as default, delay_try_seconds <= default's
}
TEST(ScaleOnDelay, FirstRatioIgnoredWithoutInitialRejection) {
    // kkt(kG, kM, 4.1, 3.0) (n = 2200, no trial: matching_trial None), first_ratio 0
    EXPECT_EQ(i.delay_stops, 1);                         // still tried
    // and later factors into the same FactorKeep behave as with first_ratio -1
}
```

- [ ] **Step 2: Run them and check they fail.** Expected: compile error, no member `scale_on_delay_first_ratio`.

- [ ] **Step 3: Implement** the branch, the env switch, and a Control comment (placeholder evidence, filled in at Step 6).

- [ ] **Step 4: Run the tests and check they pass.** Run: `"$CTEST" --test-dir build -C Release -R "ScaleOnDelay|MatchingTrial" --output-on-failure`. Expected: `100% tests passed`.

- [ ] **Step 5: Board.** Three variants, tuning split, indefinite classes, 1T, 3 rounds. Announce first: about 45 minutes.

Run: `python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json --split tuning --classes small_kkt,large_kkt,hard_indef --rounds 3 --output benchmarks/results/rescue_first_ab --binary build/benchmarks/Release/bench_replay.exe --variant off=SMF_BENCH_SCALE_ON_DELAY_FIRST=-1 --variant cap4=SMF_BENCH_SCALE_ON_DELAY_FIRST=4 --variant skip=SMF_BENCH_SCALE_ON_DELAY_FIRST=0`
then `ab_compare.py --a .../off --b .../cap4` and `--a .../off --b .../skip`, both `--metric wall`.

- [ ] **Step 6: Decide and confirm.**
  - Pick the variant with the lowest recommended-track geomean on tuning. A pick must have no accuracy loss, no case > 1.10x against `off`, and at least 1% gain over `off`. If no variant qualifies, keep -1 and record why.
  - Rerun `off` against the pick on the heldout split (same command with `--split heldout`). The pick stands if heldout shows no accuracy loss and a geomean ≤ 1.00.
  - Set the default and write the evidence into the Control comment: per-split geomeans, the rows for ncvxqp9 rec and stokes128 rec, and the worst case.

- [ ] **Step 7: Commit**

```bash
git add solver/include/smf/control.hpp solver/src/solve.cpp solver/benchmarks/bench_replay.cpp solver/tests/test_scale_on_delay.cpp
git commit -m "perf(rescue): first-try limit after an initial rejection"
```

### Task 6: Full board and write-up

**Files:**
- Create: `docs/performance/2026-11-0X-rescue-cost.md` (X = the date of the run)
- Modify: `docs/performance/2026-11-02-factor-side-delays.md` ("Open": link the new write-up from the ncvxqp9 bullet)

**Interfaces:**
- Consumes: the defaults from Tasks 4 and 5.

- [ ] **Step 1: Run the whole suite:** `"$CTEST" --test-dir build -C Release --output-on-failure`. Expected: `100% tests passed`.

- [ ] **Step 2: Board against main.** Build main's `bench_replay` into `build-main/` (a worktree at the branch point). Then run the full front-tier manifest at 1T and 8T, 3 trials each, through `tools/bench/run.py` for both binaries. Score with `python tools/bench/score.py --records <dir>/records.jsonl --manifest benchmarks/manifests/fronttier.json --split tuning` and `--split heldout`. Compare the two binaries with `ab_compare.py`. Announce first: the 8T run is multi-core and longer than 10 minutes.

Pass:
- no accuracy loss at either thread count;
- all case-tracks geomean ≤ 1.00;
- ncvxqp9 rec 1T factor and wall reported against MA97 16.0 ms (from `thread_sched_board`).

- [ ] **Step 3: Write the write-up** in the style of `2026-11-02-factor-side-delays.md`:
  - Result tables (class × split, 1T/8T);
  - the ncvxqp9 / stokes128 / PGLib rec piece tables (try, matching, factor) before and after, from Task 1's fields;
  - the edges-scanned table from Task 4 Step 5;
  - the Task 5 board and its decision;
  - Pitfalls (variants that lost);
  - Open (what is left between smf and MA97 on ncvxqp9 rec).

- [ ] **Step 4: Commit**

```bash
git add docs/performance/2026-11-0X-rescue-cost.md docs/performance/2026-11-02-factor-side-delays.md
git commit -m "docs(perf): rescue cost write-up"
```
