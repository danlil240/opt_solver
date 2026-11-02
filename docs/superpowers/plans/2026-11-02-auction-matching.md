# Auction Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the cost of the analysis' matching trial with an epsilon-scaling auction for the max-product matching. The verdicts stay the same, and so do the ordering delays and the accuracy. A second piece of work then cuts what an initial rejection pays for its candidate ordering (darcy003, mario002).

**Architecture:**
- `SymmetricMatcher` gets a second search engine, `MatchingAlgorithm::Auction`, used by the analysis trial only.
  - The constructor is unchanged: cost graph, dense split, greedy and length-2. So `initial()`, and every initial verdict, stays bitwise what it is today.
  - Phase 1 (the sparse part) runs a forward auction with epsilon-scaling on the log costs.
  - A repair pass then makes the duals exactly feasible and unmatches every pair that is not tight. The existing shortest-augmenting-path `search()` finishes those columns.
  - Phases 2-3 (dense columns) are unchanged.
  - The result is still an exact maximum-product matching whose scaling has unit matched entries and no entry above 1, because the existing exact searches finish it.
- The new engine is prototyped and gated in `bench_matching_trial` first (edges scanned, product, verdicts). Only then is it wired into `analyse()` behind `Control::matching_algorithm`, and A/B'd on the board through an environment variable.
- The initial-rejection task is separate and comes last:
  - `matching_order_from` learns to bracket the candidate's expanded flops from counts on the compressed graph;
  - it skips the expanded etree and counts when the bracket already decides the verdict.

**Tech Stack:** C++20, MSVC 2022, CMake, GoogleTest, SuiteSparse AMD (`amd_2` via `amd_order_adjacency`), Python 3.12 bench tools (`tools/bench/ab_run.py`, `ab_compare.py`, `score.py`).

**Spec:**
- The recommendation in https://claude.ai/artifact/Dvh52kmg9CwrigE8smPqUL (plan #1).
- The write-ups:
  - `docs/performance/2026-11-01-matching-search-order.md` (edges-scanned table, ncvxqp3 tail searches, what did not help);
  - `docs/performance/2026-10-31-analyse-gap.md` (pieces of the trial, initial-rejection cost, the 2% initial-scale rule);
  - `docs/performance/2026-11-02-factor-side-delays.md` (rescue matching, `set_stop_free`);
  - `docs/performance/2026-10-30-analyse-time.md` (trial paths, the `ConcurrentTrial`, the corpus survey method).

### Background (numbers this plan argues from)

1T, best rival's whole analysis against smf's matching phase alone:

| case-track | smf wall vs rival | smf matching phase vs rival analysis |
|---|---|---|
| cops_pinene_400 ipopt | 56 vs 21 ms (MA97) | 41 vs 4.2 ms |
| ncvxqp3 ipopt | 3147 vs 2171 ms | 2131 vs 639 ms |
| stokes128 ipopt | 67 vs 45 ms | 30 vs 15 ms |
| case39 ipopt | 133 vs 96 ms | 92 vs 23 ms |
| TSOPF_FS_b300_c1 ipopt | 803 vs 685 ms (SSIDS) | 422 vs 399 ms |
| darcy003 / mario002 ipopt | 409 vs 302 ms (MA57) | 211 vs 57 ms |

- If matching time is removed entirely, the 1T geomean goes from 0.80 to 0.61. Halving it gives 0.71, and the losses above 1.1x fall from 18 to 10.
- ncvxqp3 runs 16,421 searches that scan 145M edges (289x the graph). 97% of the finalized rows come off zero-reduced-cost plateaus, which every search explores again. That is the shape an auction's epsilon is made for.
- darcy003's trial: ctor 34, budgeted searches 63, initial candidate ordering 75-80 ms. The candidate ordering is paid only to reject.
- Already ruled out for the matching: stopping searches at 2% / 0.5% free, greedy-only ordering, `RealizedDelays` as default, prefetch, BFS relabel, reverse order, tie-breaks, forced METIS, and cheap rejection predictors (free fraction, off-diagonal share and r_init overlap).

## Global Constraints

- Platform:
  - Windows, MSVC 2022, CMake. `cmake` and `ctest` are not on PATH.
  - `CMAKE="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"`, and `CTEST` is the same directory's `ctest.exe`.
- Build:
  - The build directory is the worktree's `build/` (already configured).
  - Build with `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 [--target T]`. The user games on this machine.
- Tests that use CoinHSL need `SMF_HSL_LIBRARY=C:/Users/daniel/projects/opt_solver/build-deps/coinhsl-2024.5.15/CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5/bin/libhsl.dll`, with that `bin` directory on PATH.
- Threads are `std::thread` only, never OpenMP.
- What must not change:
  - `compute_symmetric_matching`, `compute_matching_scale` and the scale-on-delay rescue (`solve.cpp`: `SymmetricMatcher m(C); m.set_stop_free(...)`) stay `MatchingAlgorithm::ShortestPath` with `MatchingSearchOrder::Column`. They stay bitwise unchanged: every existing `Matching.*` test in `test_refinement.cpp` passes unedited.
  - The default `MatchingAlgorithm` argument is `ShortestPath`, so every existing caller is bitwise unchanged until Task 6 flips `Control::matching_algorithm`.
- Accuracy gate (score.py): eta <= 1e-10 and omega <= 1e-8 on every case-track that passes today. Inaccurate on every solver, and not a gap: d_pretok, turon_m, robot_arm_8k rec.
- Measuring:
  - Gate the matcher on edges scanned (`SymmetricMatcher::work()`), not on standalone matcher ms. Standalone ms drift ±20% on the Ryzen, and unpinned ncvxqp3 swings 1630-2200 ms with the same binary.
  - Boards run A and B from one binary on one machine, interleaved (`tools/bench/ab_run.py`, variants differing only by environment).
  - Boards run on the i9-12900K. Profiling and surveys run on the Ryzen 9 9950X3D.
  - The A/A noise floor is ±5% per case and ±0.3% per class.
- Benchmarks:
  - run one at a time, and check first that no `bench_replay.exe` is running;
  - run at below-normal priority;
  - announce any run longer than 10 minutes.
- The matching corpus is the 50 front-tier indefinite matrices with n >= 10000. Each task's gate builds the list the same way (Task 1, Step 1) into `benchmarks/results/auction_matching/list50.txt`. Never commit result files.
- Git:
  - Never push. Never commit CoinHSL files, the `.gitignore` change, `*.log` or `benchmarks/results/**`.
  - Commit messages end with:
    `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
    `Claude-Session: https://claude.ai/code/session_01U81susUtRyrxkAVQpHrzxU`
- Code style:
  - 4-space indent in `scaling_matching.cpp`, `ordering_matching.cpp` and `tests/`. `symbolic_analysis.cpp` keeps its Allman braces.
  - Match the surrounding comment density.
  - Every new `Control` or `Info` field, and every new constant, gets a comment with the evidence for its value.

## Review Focus

1. **A structurally singular matrix, or a sparse part with no perfect matching, under the auction.** Prices then fall without bound. The auction must stop at its bid cap and leave the remaining columns to the searches, which report the same `status()` as ShortestPath (SingularMatrix, or columns finished in phase 3). Tested in Task 3: `AuctionMatching.StructurallySingularIsReported` and `AuctionMatching.InfeasibleSparsePartFinishesInPhase3`.
2. **A budget pause mid-auction, then resume.** This happens with serial budgets and with the `ConcurrentTrial`'s one-pass slices. The result must be bitwise the one-shot result. Tested in Task 3 (`AuctionMatching.ResumedAdvanceIsOneShot`, pause every 50 edges) and Task 5 (`MatchingTrial.ConcurrentTrialIsTheSerialOne` run under Auction too).
3. **A rejecter changes trial path.** A cheaper matcher can finish inside the 2-pass budget, so `RejectedInitial` becomes `Exact` with a rejection. That silently drops the initial-scale rule `kInitialScaleMaxFree`: stokes128 rec has 1.0% free, and its delays go 17k back to 54k, wall 65 → 159 ms. Checked in Task 5: Step 5 compares paths per matrix, and Step 6 is the conditional fix with its test.
4. **Magnitudes spanning hundreds of decades.** Costs reach ~1500 in log units, while the scale's log is clamped at ±340 in `State::result`. epsilon0 is derived from the largest cost, and one huge cost must not make every phase a price war. Tested in Task 3: `AuctionMatching.WideMagnitudeRange` (spread 40 decades), with the same product as ShortestPath and the same scaling bounds.
5. **Columns with a single sparse entry, explicit zeros and duplicate entries** (input that is not cleaned). A single-entry column has no second-best row, and the bid rule must still terminate. Tested in Task 3: `AuctionMatching.SingleEntryColumnsAndUncleanedInput`.

---

### Task 1: Survey baseline: degree-order work, product and timing per matrix

The trial's matcher runs in Degree order on analyse()'s cleaned copy. `bench_matching_trial` today measures `work_x` with a Column-order matcher on the raw matrix. This task makes the survey report what the trial actually runs, and records the ShortestPath baseline every later gate compares against.

**Files:**
- Modify: `solver/benchmarks/bench_matching_trial.cpp` (header comment, the `--list` survey loop, the usage string)

**Interfaces:**
- Produces:
  - a `--matcher ssp|auction` flag. Task 1 accepts only `ssp`; Task 4 enables `auction`.
  - new survey columns: `work_deg`, the edges scanned by `SymmetricMatcher(A, MatchingSearchOrder::Degree)` run to the end, in passes of 2 x lower entries; `logprod_hash`, an fnv hash of the final `col_to_row`; `logprod`, `sum_j log|a(col_to_row[j], j)|` printed with `%.12e`; and `mtd_ms`, the median ms of the Degree-order matcher run to the end.
  - a totals line that adds `edges_deg <sum of work()>`.

- [ ] **Step 1: Build the corpus list**

Run:
```bash
mkdir -p benchmarks/results/auction_matching
python -c "import json;m=json.load(open('benchmarks/manifests/fronttier.json'));s=sorted({(c['matrix_id'],c['mtx_path']) for c in m['cases'] if c['matrix_type']=='indefinite' and c['n']>=10000});open('benchmarks/results/auction_matching/list50.txt','w').write(''.join(f'{i} {p}\n' for i,p in s));print(len(s))"
```
Expected: `50`.

- [ ] **Step 2: Add the flag and columns**

`--matcher` is parsed like `--ordering`. Any value but `ssp` prints the usage and returns 2. The Degree-order matcher is timed in the same `trials` loop as `mt_ms` and hashed into `all_hash`. `all_hash` therefore changes once, in this task, by design.

- [ ] **Step 3: Build and run the baseline**

Run: `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 --target bench_matching_trial`, then
`build/benchmarks/Release/bench_matching_trial.exe --list benchmarks/results/auction_matching/list50.txt --trials 3 --matcher ssp > benchmarks/results/auction_matching/ssp_survey.txt`.
Expected:
- 50 rows and `0 differ from the exact trial`;
- `work_deg` for case39 near 2.5M edges in passes, TSOPF_FS_b300_c3 near 47M and ncvxqp3 near 147M (the write-up's degree-order column);
- `edges_deg` total near 553M.

- [ ] **Step 4: Commit**

```bash
git add solver/benchmarks/bench_matching_trial.cpp
git commit -m "bench_matching_trial: report the trial's degree-order matcher work and product"
```

---

### Task 2: Auction kernel on the sparse part

A free function, separate from `SymmetricMatcher`, so it can be tested on its own invariants. The cost graph and row state move to an internal header that both files share.

**Files:**
- Create: `solver/src/matching_internal.hpp` (move `CostGraph`, `build_cost_graph`, `RowState` and `kInf` out of `scaling_matching.cpp`'s anonymous namespace into `namespace smf::detail`; no behavior change)
- Create: `solver/src/auction_matching.hpp`, `solver/src/auction_matching.cpp`
- Modify: `solver/include/smf/scaling_matching.hpp` (`AuctionOptions` only)
- Modify: `solver/src/scaling_matching.cpp` (include `matching_internal.hpp`, use `detail::`), `solver/CMakeLists.txt` (add `src/auction_matching.cpp` to the `smf` sources)
- Test: `solver/tests/test_auction_matching.cpp` (new), `solver/tests/CMakeLists.txt` (`add_executable(test_auction_matching ...)`, linked like `test_matching_trial`, and `add_test(NAME AuctionMatching COMMAND test_auction_matching)`)

**Interfaces:**
- Produces: `AuctionOptions` in the public `solver/include/smf/scaling_matching.hpp` (namespace `smf`, documented as the auction's tuning, because Task 3's constructor and Task 4's tool take it), and the kernel in `auction_matching.hpp` (namespace `smf::detail`; tests include it as `"../src/auction_matching.hpp"`, as `test_indexed_min_heap` does):
```cpp
// scaling_matching.hpp
struct AuctionOptions {
    double eps_factor = 5.0;   // epsilon divides by this per scaling phase
    double eps_final  = 1e-3;  // last phase's epsilon, in log units
    bool   warm       = true;  // start from the constructor's duals and greedy matching
    double bid_cap    = 50.0;  // max bids per phase, in units of the sparse part's columns
};
// auction_matching.hpp
enum class AuctionStatus { Done, Paused, Capped };
struct AuctionState {        // resumable: everything advance needs to continue
    double eps = 0.0;
    int    phase = -1;       // -1: not started
    std::vector<int> queue;  // free sparse columns, FIFO
    std::size_t head = 0;
    long long bids_in_phase = 0;
};
// Runs bids until every sparse column is matched in the last phase (Done),
// work >= work_budget at a bid boundary (Paused; work_budget < 0: none),
// or a phase exceeds opt.bid_cap x sparse columns bids (Capped: the
// columns left free stay free). Dense rows/columns (dense[i] != 0) are
// invisible. u: row duals, v: column duals, row_match / col_to_row: the
// matching, all updated in place; work += every scanned edge.
AuctionStatus auction_advance(const CostGraph &g, const std::uint8_t *dense,
                              std::vector<RowState> &rs, std::vector<double> &v,
                              std::vector<int> &row_match, std::vector<Int> &col_to_row,
                              const std::vector<int> &first_queue, const AuctionOptions &opt,
                              AuctionState &st, long long &work, long long work_budget);
```
`first_queue` holds the free sparse columns in the trial's search order (Degree). It is read when `st.phase == -1`.

- [ ] **Step 1: Write the failing tests**

All tests build their graph with `detail::build_cost_graph`, starting from the constructor duals: u_i = min_j c_ij over the sparse part, v_j = min_i(c_ij - u_i). Shared checks:
- `eps_cs(g, rs, v, col_to_row, eps)` asserts `c_ij - u_i - v_j >= -eps - 1e-12` on every sparse edge, for v_j recomputed as min_i(c_ij - u_i). It also asserts that each matched edge is within eps of its column's minimum.
- `log_product(A, col_to_row)` is the sum of log|a| over matched entries.

```cpp
TEST(AuctionMatching, KernelReachesEpsCsAndPerfectMatching)  // opf_like_kkt(3000, 1200, seeds 1..3)
    // AuctionStatus::Done, every column matched, eps_cs(..., opt.eps_final) holds,
    // and log_product within n * eps_final of compute_symmetric_matching's.
TEST(AuctionMatching, KernelOptimalAgainstBruteForce)        // 60 random n = 3..7 matrices, as Matching.OptimalAgainstBruteForce
    // with eps_final = 1e-9 / n: product equals brute force within 1e-9 * (1 + |best|);
    // a structurally singular one ends Capped, never Done.
TEST(AuctionMatching, KernelPausesAtBidBoundaries)           // budget = work + 50 in a loop
    // each Paused call leaves work >= budget; the final state (u, v, matching, work) equals the one-shot run bitwise.
TEST(AuctionMatching, KernelColdAndWarmStartsAgreeOnProduct) // warm = false vs true on the seeds above
    // both Done, products within n * eps_final of each other.
```

- [ ] **Step 2: Run them to verify they fail**

Run: `nice -n 10 "$CMAKE" --build build --config Release --parallel 4 --target test_auction_matching`.
Expected: compile error, because `auction_matching.hpp` does not exist.

- [ ] **Step 3: Implement `auction_advance` in `solver/src/auction_matching.cpp`**

It is a forward (Gauss-Seidel) auction in min-cost form. Columns bid for rows, and a row's price is -u_i. This is the bid rule:

```
column j (front of queue) scans its sparse rows i (work += its full edge count):
    b1 = min_i (c_ij - u_i) at row i1;  b2 = second smallest
    (only one sparse row: b2 = b1 + max(eps, cost_range), cost_range = max - min sparse cost)
    v_j    = b2
    u[i1]  = c_{i1 j} - b2 - eps      // row i1's price rises by (b2 - b1) + eps
    if row_match[i1] = k >= 0: col_to_row[k] = -1, push k to the back of the queue
    row_match[i1] = j, col_to_row[j] = i1
```

Phases and starting values:
- Cold start: unmatch everything, then use the constructor duals and eps0 = max sparse cost / eps_factor.
- Warm start: keep the greedy matching, which satisfies 0-CS. eps0 = max sparse cost / eps_factor as well. Task 4 tunes both choices.
- Phase change: eps = max(eps / eps_factor, eps_final). One pass over the matched sparse columns (work += nnz) unmatches and queues every column whose matched edge is more than eps above its column's minimum.
- Ties in b1 go to the lowest row position in the column (deterministic).
- `bids_in_phase` counts against `bid_cap * (sparse columns)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `"$CTEST" --test-dir build -C Release -R "AuctionMatching|Matching$|MatchingTrial" --output-on-failure`.
Expected: `100% tests passed`. The `Matching` tests show the moved cost graph is unchanged.

- [ ] **Step 5: Commit**

```bash
git add solver/src/matching_internal.hpp solver/src/auction_matching.hpp solver/src/auction_matching.cpp solver/include/smf/scaling_matching.hpp \
        solver/src/scaling_matching.cpp solver/CMakeLists.txt solver/tests/test_auction_matching.cpp solver/tests/CMakeLists.txt
git commit -m "matching: epsilon-scaling auction kernel for the sparse part"
```

---

### Task 3: `SymmetricMatcher` with `MatchingAlgorithm::Auction` and the exact repair

**Files:**
- Modify: `solver/include/smf/types.hpp` (new enum next to `MatchingTrial`), `solver/include/smf/scaling_matching.hpp`, `solver/src/scaling_matching.cpp` (`State`, constructor, `advance`, new `State::repair()`)
- Test: `solver/tests/test_auction_matching.cpp`

**Interfaces:**
- Consumes: `AuctionOptions` (Task 2, public), and `detail::auction_advance`, `AuctionState`, `AuctionStatus` (Task 2).
- Produces:
```cpp
// types.hpp
enum class MatchingAlgorithm { ShortestPath = 0, Auction = 1 };
// scaling_matching.hpp
explicit SymmetricMatcher(const CscLower &A, MatchingSearchOrder order = MatchingSearchOrder::Column,
                          MatchingAlgorithm alg = MatchingAlgorithm::ShortestPath,
                          const AuctionOptions *auction = nullptr);  // nullptr: the defaults
int repaired() const;  // Auction: sparse columns the repair handed to search(); 0 otherwise
```
- Precondition, documented on `set_stop_free`: ShortestPath only. Under Auction it asserts in debug and is ignored in release.
- `work()` counts the auction's bids, its phase passes, the repair pass and the repair searches.

- [ ] **Step 1: Write the failing tests**

Helpers:
- `same_optimum(A, a, b)`: both statuses Success, matched_rank n, and log products equal within `1e-9 * |p| + 1e-9`.
- `scaling_bounds(A, r)`: every |s_i a_ij s_j| <= 1 + 1e-10, and every matched one within 1e-10 of 1 (as in `MatchingTrial.DegreeSearchOrderIsAMaximumProductMatching`).

```cpp
TEST(AuctionMatching, SameOptimumAsShortestPathOnKkt)
    // scaled_opf_kkt(3000, 1200, seeds 1..3, spread 4.0), and a boyd1-like KKT with 3 dense rows (phases 1-3):
    // SymmetricMatcher(A, Degree, Auction) advance() true; same_optimum vs compute_symmetric_matching; scaling_bounds.
TEST(AuctionMatching, InitialIsTheShortestPathInitial)
    // initial().col_to_row and .scale bitwise equal to SymmetricMatcher(A, Degree).initial(); diagonal() equal.
TEST(AuctionMatching, ResumedAdvanceIsOneShot)
    // for (budget = 0; !m.advance(budget); budget = m.work() + 50) ++pauses;  pauses > 1;
    // current() bitwise equal to a one-shot Auction matcher (col_to_row, row_to_col, scale), work() equal.
TEST(AuctionMatching, StructurallySingularIsReported)
    // the matrices of Matching.StructurallySingular and Matching.StructurallySingularFailsAfterSearches:
    // status() equals ShortestPath's (SingularMatrix) and advance() returns true.
TEST(AuctionMatching, InfeasibleSparsePartFinishesInPhase3)
    // a KKT whose sparse part has no perfect matching (two sparse columns sharing a single sparse row,
    // each also in a dense row): Success, same_optimum, scaling_bounds.
TEST(AuctionMatching, WideMagnitudeRange)
    // scaled_opf_kkt(3000, 1200, seed 4, spread 40.0): same_optimum, scaling_bounds, and
    // work() <= 4 x the Degree ShortestPath work (no price war).
TEST(AuctionMatching, SingleEntryColumnsAndUncleanedInput)
    // columns with one off-diagonal entry; duplicates summed and explicit zeros (as Matching.ExplicitZerosKeepLaterEntries):
    // same_optimum vs compute_symmetric_matching.
TEST(AuctionMatching, EmptyAndTiny)
    // n = 0 and diagonal5(): status Success, current() equals ShortestPath's bitwise.
```

- [ ] **Step 2: Run them to verify they fail**

Build `test_auction_matching`. Expected: compile error, because there is no `MatchingAlgorithm`.

- [ ] **Step 3: Implement**

In `advance`, phase 1 under Auction:
1. The first call fills `search_order` as today, and Degree order applies.
2. `auction_advance(..., first_queue = search_order, ..., work_budget)`. On `Paused` it returns false.
3. On `Done` or `Capped`, `State::repair()` runs once:
   - v_j = min over sparse rows of (c_ij - u_i) for every sparse column (work += sparse nnz). The duals are now exactly feasible.
   - Every matched pair with `c_ij - u_i - v_j > 0` is unmatched.
   - The columns left free are listed in `search_order` (Degree order), and `next_col` is reset.
4. The existing phase-1 search loop then runs on them, honoring the same budget, followed by phases 2-3 unchanged.

The repair runs at most once, tracked by a `repaired_` flag. Pausing inside the repair searches resumes in the loop as today.

`State::result` is unchanged. It re-derives v on the matched (tight) edges, so the scaling guarantees follow from the exact searches.

- [ ] **Step 4: Run the tests**

Run: `"$CTEST" --test-dir build -C Release -R "AuctionMatching|Matching$|MatchingTrial|Refinement|Scaling" --output-on-failure`.
Expected: `100% tests passed`.

- [ ] **Step 5: Commit**

```bash
git add solver/include/smf/types.hpp solver/include/smf/scaling_matching.hpp solver/src/scaling_matching.cpp solver/tests/test_auction_matching.cpp
git commit -m "matching: SymmetricMatcher auction engine with exact shortest-path repair"
```

---

### Task 4: Prototype in `bench_matching_trial`: tune, then gates G1 (optimum) and G2 (edges)

**Files:**
- Modify: `solver/benchmarks/bench_matching_trial.cpp`

**Interfaces:**
- Consumes: `SymmetricMatcher(A, Degree, MatchingAlgorithm::Auction, &opt)` and `repaired()` (Task 3).
- Produces:
  - `--matcher auction` with `--auction-factor f --auction-eps e --auction-cold`, which fill an `AuctionOptions`;
  - survey columns `work_auc` (passes), `repair` (`repaired()`), `mta_ms`, and `prod_ok` (`yes` when the log product equals the Degree ShortestPath one within `1e-9 * |p| + 1e-9` and `scaling_bounds` hold);
  - a totals line with `edges_auc`, `prod_fail <count>`, and the auction's worst `work_auc / work_deg`.
- Until Task 5, `--matcher` affects only the standalone matcher columns, not the three `analyse()` calls.

- [ ] **Step 1: Add the flags and columns**

`prod_ok` reuses the product and bounds checks written for the tests, as local functions in the tool.

- [ ] **Step 2: Sweep the options on the corpus**

Run, on the Ryzen, one configuration at a time (with `--trials 1`, since work is deterministic):
```bash
for f in 4 7 10; do for e in 1e-2 1e-3 1e-4 1e-6; do for c in "" --auction-cold; do
  build/benchmarks/Release/bench_matching_trial.exe --list benchmarks/results/auction_matching/list50.txt --trials 1 \
    --matcher auction --auction-factor $f --auction-eps $e $c > benchmarks/results/auction_matching/sweep_${f}_${e}${c}.txt
done; done; done
```
Expected: every file reports `prod_fail 0`.

- [ ] **Step 3: Pick the defaults and record them**

The winner is the configuration with the lowest `edges_auc`. Ties within 2% go to the one with the lower ncvxqp3 `work_auc`. Write the winner into the `AuctionOptions` defaults in `scaling_matching.hpp`, with the evidence comment: total edges, ncvxqp3, case39, pinene and TSOPF_FS_b300_c3 against the `work_deg` baseline.

- [ ] **Step 4: Gate G1 + G2 (the kill point)**

Run: the survey with `--matcher auction --trials 3` → `benchmarks/results/auction_matching/auction_survey.txt`, and compare it with `ssp_survey.txt` from Task 1.

PASS requires all of:
- **G1:** `prod_fail 0` on all 50 matrices, and the same status as ShortestPath on each.
- **G2:**
  - `edges_auc` <= 0.6 x `edges_deg` in total;
  - ncvxqp3 `work_auc` <= 0.5 x its `work_deg`;
  - no matrix whose `work_deg` exceeds 1 pass scans more than 1.5x its `work_deg`;
  - the `mta_ms` geomean over the 50 is <= 0.8 x the `mtd_ms` geomean. This one is checked on pinned runs (`start /affinity 0xFFFF`, CCD 0), because ms drift.

**If G1 or G2 fails:**
- record the sweep and the failure in `docs/performance/2026-11-xx-auction-matching.md`: the tail searches, `repair` counts, and where the edges go;
- commit only the tool and the write-up;
- skip Tasks 5-6 and go to Task 7, which does not depend on the auction.

A large `repair` count with a small `work_auc` points at the repair, not the auction. Record it, and note Goldberg-Kennedy cost-scaling push-relabel (CSA) as the follow-up rather than switching algorithms inside this plan.

- [ ] **Step 5: Commit**

```bash
git add solver/benchmarks/bench_matching_trial.cpp solver/include/smf/scaling_matching.hpp
git commit -m "bench_matching_trial: auction matcher prototype, tuned defaults"
```

---

### Task 5: Wire the auction into the analysis trial; gate G3 (verdicts and paths)

**Files:**
- Modify:
  - `solver/include/smf/control.hpp` (new field after `matching_search_budget`);
  - `solver/src/symbolic_analysis.cpp` (the serial `std::make_unique<SymmetricMatcher>(C, MatchingSearchOrder::Degree)`, and `ConcurrentTrial`, which takes the algorithm as a constructor argument and passes it to `SymmetricMatcher matcher(C_, MatchingSearchOrder::Degree)` in `run()`);
  - `solver/benchmarks/bench_replay.cpp` (environment variable, next to `SMF_BENCH_AMALGAMATION`);
  - `solver/benchmarks/bench_matching_trial.cpp` (`--matcher` also sets `Control::matching_algorithm` on the three analyses).
- Test: `solver/tests/test_matching_trial.cpp`

**Interfaces:**
- Consumes: `MatchingAlgorithm` (Task 3).
- Produces:
  - `Control::matching_algorithm`, of type `MatchingAlgorithm`, default `ShortestPath` until Task 6. It applies to analyse()'s auto matching trial only.
  - `SMF_BENCH_MATCHER=ssp|auction` in `bench_replay`.
  - A `ConcurrentTrial` constructor with a `MatchingAlgorithm alg` parameter, placed after `expected_refactors`.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(MatchingTrial, AuctionTrialKeepsVerdicts)
    // for each of: scaled_opf_kkt(3000, 1200, 1, 4.0), the matrices of ResumesOnSyntheticKkt, RejectsOnInitialMatchingPastBudget
    // and DiagonalMatchingSkipsTheTrial, at budgets -1, 0 and the default:
    // Info::ordering_used and AnalysisKeep::matching_rejected equal between ShortestPath and Auction;
    // matching_trial equal, except Exact <-> ResumedExact; matching_initial_ratio bitwise equal when both set.
TEST(MatchingTrial, AuctionConcurrentTrialIsTheSerialOne)
    // ConcurrentTrialIsTheSerialOne's body with c.matching_algorithm = MatchingAlgorithm::Auction:
    // 1 vs 4 threads give identical perm, matching scale, rejected flag and verdict on every budget path.
```

- [ ] **Step 2: Run them to verify they fail**

Build `test_matching_trial`. Expected: compile error, because `matching_algorithm` is not a member of `Control`.

- [ ] **Step 3: Implement the field and the plumbing**

Its comment cites the G1/G2 numbers from Task 4.

- [ ] **Step 4: Run the tests**

Run: the full `"$CTEST" --test-dir build -C Release -j 4 --output-on-failure`.
Expected: `100% tests passed`.

- [ ] **Step 5: Gate G3: verdicts, paths and ratios on the corpus**

Run: `bench_matching_trial --list .../list50.txt --trials 3 --matcher auction > .../auction_trial.txt`, and the same with `--ordering 1 --refactors 10` (the IPOPT plugin's configuration) for both matchers.

PASS requires:
- `0 differ from the exact trial` on both matchers;
- per matrix, the `default` verdict and `ordD` (ordering used) equal to the ssp run, except `exact` <-> `resumed`;
- every change in `perm_hash` listed with its `r_exact` on both sides.

Every matrix whose path moves from `rej-init` to `exact` (rejected) is listed as well, together with its `free0` column.

- [ ] **Step 6: Only if Step 5 lists a moved rejecter with `free0` <= 0.02 (stokes128-like)**

Extend the `kInitialScaleMaxFree` rule in `symbolic_analysis.cpp` (the block at `initial_free_share <= kInitialScaleMaxFree && control.pivot_u >= kWeakPivotRatio`):
- an exact rejection whose *initial* matching left at most 2% free also applies the initial matching's scale as a partial scale;
- `initial_free_share` is set from `initial().matched_rank` on every rejection path, serial and `ConcurrentTrial`.

Test:
```cpp
TEST(MatchingTrial, ExactRejectionKeepsTheInitialScaleRule)
    // a KKT whose initial matching leaves <= 2% free and whose exact trial rejects (budget -1), u = 0.01:
    // keep->matching_scale_partial true and keep->matching_scale equal to the initial matching's scale;
    // at u = 1e-8: matching_scale empty.
```
Then run the full ctest. cops_pinene_400 rec, whose full exact scale broke it, must stay accurate on the Task 6 board.

- [ ] **Step 7: Commit**

```bash
git add solver/include/smf/control.hpp solver/src/symbolic_analysis.cpp solver/benchmarks/bench_replay.cpp \
        solver/benchmarks/bench_matching_trial.cpp solver/tests/test_matching_trial.cpp
git commit -m "analyse: Control::matching_algorithm selects the trial's matcher (auction behind a switch)"
```

---

### Task 6: Board gate G4 (delays, accuracy, wall), default flip, write-up

**Files:**
- Modify: `solver/include/smf/control.hpp` (default `MatchingAlgorithm::Auction`, if G4 passes), `docs/roadmap.md` (one line under the current horizon's analysis levers)
- Create: `docs/performance/2026-11-xx-auction-matching.md` (the date of the run)
- Test: existing suites

**Interfaces:**
- Consumes: `SMF_BENCH_MATCHER` (Task 5).

- [ ] **Step 1: 1-thread board, interleaved, on the 12900K**

Run (announce it first: about 1 hour):
```bash
python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json --split all \
  --classes small_kkt,large_kkt,hard_indef --rounds 3 --threads 1 \
  --output benchmarks/results/auction_matching/board_t1 \
  --binary build/benchmarks/Release/bench_replay.exe \
  --variant ssp=SMF_BENCH_MATCHER=ssp --variant auc=SMF_BENCH_MATCHER=auction
python tools/bench/ab_compare.py --a benchmarks/results/auction_matching/board_t1/ssp \
  --b benchmarks/results/auction_matching/board_t1/auc --manifest benchmarks/manifests/fronttier.json \
  --metric wall --output benchmarks/results/auction_matching/board_t1/wall.md
```
Also run it with `--metric analyse`.

- [ ] **Step 2: 8-thread board on the trial-heavy cases**

Same commands with `--threads 8 --cases ncvxqp3,ncvxqp5,ncvxqp7,cvxqp3,case39,cops_pinene_400,stokes128,darcy003,mario002,TSOPF_FS_b300_c1,TSOPF_FS_b300_c3,pglib_opf_case2000_goc__sad`.

- [ ] **Step 3: Gate G4**

PASS requires all of:
- **Accuracy:** no case-track accurate under ssp is inaccurate under auction (`ab_compare` lists none).
- **Delays:** per case-track, factor delays under auction within +10% of ssp, or the case is listed with its `r_exact` and wall. A listed case whose wall loses more than 5% fails the gate.
- **Wall:**
  - all indefinite case-tracks at 1T: geomean <= 0.97;
  - hard_indef, both splits: <= 0.95;
  - no case-track > 1.10x after a 3-round rerun of that case alone;
  - at 8T, geomean <= 0.97 on the listed cases.
- **Targets read off the analyse metric**, reported but not gated: cops_pinene_400 ipopt, ncvxqp3 ipopt, stokes128, case39 and TSOPF_FS_b300_c1 matching phase against the Background table.

- [ ] **Step 4: If G4 passes, flip the default**

Set `Control::matching_algorithm = MatchingAlgorithm::Auction`. Its comment carries the G4 numbers. Run the full ctest: `100% tests passed`. Tests that compare against the trial's ShortestPath behavior set it explicitly.

If G4 fails, the default stays `ShortestPath` and the write-up records why.

- [ ] **Step 5: Write-up and roadmap**

Follow the house format of `docs/performance/2026-11-01-matching-search-order.md`:
- Result table (class × split × track, 1T and 8T);
- edges-scanned table (ssp degree vs auction, the same matrices as that write-up);
- the tuning sweep;
- the path changes from G3;
- pitfalls;
- a Measuring section.

In `docs/roadmap.md`, one line: the result, and the link.

- [ ] **Step 6: Commit**

```bash
git add solver/include/smf/control.hpp docs/performance/2026-11-xx-auction-matching.md docs/roadmap.md
git commit -m "analyse: auction matcher for the matching trial by default; write-up"
```

---

### Task 7 (later, independent of Tasks 2-6): The initial-rejection cost on darcy003 / mario002

An initial rejection builds the initial matching's compressed AMD ordering and its exact expanded flops (etree, postorder and column counts on A), and then throws them away (`initial_passes` in `symbolic_analysis.cpp`). The verdict needs only a comparison:
- reject when flops > 2.4 fl0 (`kMatchingInitialRejectRatio * fl0`) and flops < dear_flops;
- resume otherwise.

A bracket LB <= flops <= UB from column counts on the *compressed* graph settles that comparison without the expanded counts, whenever LB > 2.4 fl0 and UB < dear_flops. The verdict is then identical by construction.

Why the bracket holds:
- A pair (a, b) is eliminated a then b, and the a-b entry is nonzero (it is the matched entry).
- So b's column is the expansion of node J's compressed column structure: each node in it contributes 1 or 2 rows.
- With cc_J the unweighted compressed count of node J (diagonal included):

  LB = Σ_J cc_J² + 4 · (pairs)
  UB = Σ_J (2cc_J − 1)² + Σ_pairs (2cc_J)²

Check this bound on random cases before relying on it (Step 3's test does).

**Files:**
- Modify:
  - `solver/include/smf/ordering.hpp` (`MatchingOrderStats`, a new `FlopsCut`, a parameter of `matching_order_from`);
  - `solver/src/ordering_matching.cpp`;
  - `solver/src/symbolic_analysis.cpp` (`initial_passes`);
  - `solver/include/smf/info.hpp` (one field), `solver/include/smf/control.hpp` (one switch);
  - `solver/benchmarks/bench_replay.cpp` (`SMF_BENCH_MATCHING_CUT`), `solver/benchmarks/bench_matching_trial.cpp` (`--pieces` columns).
- Test: `solver/tests/test_matching_trial.cpp`

**Interfaces:**
- Produces:
```cpp
// ordering.hpp
struct FlopsCut {
    double reject_above;       // decide "reject" when LB exceeds this ...
    double resume_at_or_above; // ... and UB stays below this
};
struct MatchingOrderStats {    // added fields
    double flops_lb = 0.0, flops_ub = 0.0;  // set when a FlopsCut was given (how == Amd)
    bool   flops_bounded = false;           // flops == flops_lb: expanded counts skipped, etree empty
    double seconds_graph = 0.0, seconds_order = 0.0, seconds_flops = 0.0;  // compressed graph / AMD / expanded counts
};
bool matching_order_from(..., const Int *a_xadj = nullptr, const Int *a_adj = nullptr,
                         const FlopsCut *cut = nullptr);   // new trailing parameter; ignored unless how == Amd
// info.hpp
bool matching_initial_bounded = false;  // matching_initial_ratio is then LB / base flops (a lower bound)
// control.hpp
bool matching_flops_cut = true;         // initial verdict through FlopsCut; false: today's exact counts
```

- [ ] **Step 1: Measure where the candidate ordering's time goes**

Add the three `seconds_*` fields (`steady_clock` laps in `matching_order_from`, only when `stats != nullptr`). Add `--pieces` columns `oi_graph`, `oi_amdc` and `oi_cnt`, split from `oi_amd`. Then run:
`bench_matching_trial --list .../list50.txt --pieces --trials 3 --case darcy003` (and mario002, stokes128, pglib_opf_case2000_goc__sad, turon_m).
Expected: the split adds up to `oi_amd` within 5%.

- [ ] **Step 2: Decision point**

If `oi_cnt` is below 25% of `oi_amd` on both darcy003 and mario002, the bracket cannot reach the gate below:
- record the split in the write-up (Task 6's document, or a new `docs/performance/2026-11-xx-initial-rejection.md` if Task 6 was skipped), as the measured reason;
- commit only the stats fields and the tool columns;
- stop.

Otherwise continue.

- [ ] **Step 3: Write the failing tests**

```cpp
TEST(MatchingTrial, CompressedFlopBoundsBracketExact)
    // opf_like_kkt(3000, 1200, seeds 1..5) and scaled_opf_kkt(.., 4.0), both initial() and current() matchings,
    // cut {inf, inf}: stats.flops_lb <= stats.flops <= stats.flops_ub, flops_bounded false.
TEST(MatchingTrial, FlopsCutSkipsCountsOnlyWhenTheBracketDecides)
    // cut {0.5 * lb, inf}: flops_bounded true, flops == flops_lb, etree empty, perm equal to the uncut call's;
    // cut {2 * ub, inf} and cut {0.5 * lb, 0.5 * ub}: flops_bounded false, flops, perm, etree bitwise the uncut call's.
TEST(MatchingTrial, BoundedInitialRejectionIsTheExactVerdict)
    // RejectsOnInitialMatchingPastBudget's matrix: with matching_flops_cut true and false,
    // matching_trial RejectedInitial on both, the same AnalysisKeep perm and matching scale;
    // matching_initial_bounded set only on the cut run.
```

- [ ] **Step 4: Run them to verify they fail**

Build `test_matching_trial`. Expected: compile error, because there is no `FlopsCut`.

- [ ] **Step 5: Implement**

In `matching_order_from`, with `cut != nullptr && how == CompressedOrder::Amd`:
- after AMD on the compressed graph, run `etree_from_adjacency`, `etree_postorder` and `filled_column_counts_from_adjacency` on `cxadj` / `cadj` with `node_perm`;
- form LB and UB as above;
- skip `predicted_flops` when `LB > cut->reject_above && UB < cut->resume_at_or_above`.

In `initial_passes`, when `control.matching_flops_cut` is set:
- pass `FlopsCut{kMatchingInitialRejectRatio * fl0, dear_flops}`;
- a bounded result rejects, with `info.matching_initial_ratio = stats.flops / fl0` and `info.matching_initial_bounded = true`.

Both the serial and the concurrent paths go through `initial_passes`, so both are covered.

`bench_replay` maps `SMF_BENCH_MATCHING_CUT=0|1` to `matching_flops_cut`.

- [ ] **Step 6: Run the tests**

Run: the full ctest. Expected: `100% tests passed`.

- [ ] **Step 7: Gate**

1. Corpus survey: `bench_matching_trial --list .../list50.txt --trials 3`, before and after.
   - The `perm_hash` column, the three verdict columns and `ordD` must be identical on all 50. `all_hash` changes by design, because the initial ratio is now a bound on bounded matrices.
   - On every non-bounded matrix, the matching phase (`mm_def`) must stay within +3%.
2. A/B on the 12900K, interleaved, both tracks, 1T and 8T:
   `ab_run.py --cases darcy003,mario002,stokes128,pglib_opf_case2000_goc__sad,pglib_opf_case1354_pegase,turon_m --rounds 5 --variant off=SMF_BENCH_MATCHING_CUT=0 --variant on=SMF_BENCH_MATCHING_CUT=1`, then `ab_compare.py --metric analyse` and `--metric wall`.

PASS requires:
- darcy003 and mario002 analyse <= 0.85x on both tracks at 1T;
- no case-track worse than 1.03x on analyse;
- identical delays and accuracy (the factors are bitwise the same, since the ordering and scale are unchanged).

- [ ] **Step 8: Commit**

```bash
git add solver/include/smf/ordering.hpp solver/src/ordering_matching.cpp solver/src/symbolic_analysis.cpp \
        solver/include/smf/info.hpp solver/include/smf/control.hpp solver/benchmarks/bench_replay.cpp \
        solver/benchmarks/bench_matching_trial.cpp solver/tests/test_matching_trial.cpp
git commit -m "analyse: decide initial rejections from compressed-count flop bounds"
```

Add the A/B table and the split from Step 1 to the write-up, and commit it with the code.
