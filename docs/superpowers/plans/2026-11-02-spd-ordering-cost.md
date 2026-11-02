# SPD Ordering Cost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the cost of nested dissection on large SPD matrices without giving up its fill. On cfd2, smf's ordering phase alone (848 ms) costs more than PARDISO's whole analysis (540 ms). bmwcra_1 and nd24k have the same problem. Fill and flops must not get worse, and the factor must not get slower.

**Architecture:** The plan starts with measurement. The SPD `AutoParallel` path in `symbolic_analysis.cpp` (step 5) always runs AMD first. When AMD's predicted flops reach `kMetisFlopsPerRow` per row, it then runs one of two orderings:
- at 1 thread, serial `METIS_NodeND` on a freshly built uncompressed adjacency (`build_symmetric_adjacency`);
- at more threads, `parallel_nd_order`. That is closed-adjacency compression plus METIS separators to depth 4, with leaves ordered concurrently. Every METIS call, though, goes through one process-wide mutex (`metis_mutex()` in `ordering_metis.cpp`), because the linked conda-forge METIS 5.2.1 aborts when called concurrently. So the "parallel" ND only overlaps its AMD leaves and graph work.

The plan has five levers, each behind a `Control` knob so that one binary can A/B them:
1. per-call METIS options (fewer refinement passes, cheaper coarsening);
2. compression in front of the 1-thread path, by running `parallel_nd_order` at every thread count;
3. a thread-safe METIS build, which makes the leaf and same-level METIS calls truly concurrent;
4. overlapping or skipping the AMD pre-pass when the profile shows it matters;
5. a policy commit that keeps only what passed the gates.

**Tech Stack:** C++20, MSVC 2022 (CMake, `std::thread`), METIS 5.2.1 + GKlib, SuiteSparse AMD, GoogleTest, Python 3.12 bench tools (`tools/bench/ab_run.py`, `ab_compare.py`, `score.py`).

**Spec:**
- the recommendation in `https://claude.ai/artifact/Dvh52kmg9CwrigE8smPqUL`;
- `docs/performance/2026-10-25-large-scale-benchmark.md` ("Serial METIS dominates analysis on bmwcra_1");
- `docs/performance/2026-10-30-analyse-time.md` and `docs/performance/2026-10-31-analyse-gap.md` (phase timers, forced-METIS cost);
- `docs/performance/2026-10-29-path-to-number-1.md` (cfd2/bmwcra_1 rows);
- commit `b948946` (the parallel ND, its flops gate, and why METIS is serialised).

## Background

Standing on 2026-11-02, 1 thread, recommended track: cfd2 (SPD) loses 1.16x to PARDISO (wall 1493 vs 1289 ms). smf's METIS ordering takes 848 ms; PARDISO's whole analysis takes 540 ms. At 8/16 threads nd24k loses 1.4-1.6x to SSIDS. Most of that gap is the factor (plan #4), but analysis is part of it too. Earlier measurements:
- bmwcra_1 AutoParallel analysis was 0.842 s at 1T and 0.614 s at 2T (`b948946`), with METIS about 0.6 of 0.85 s.
- Compression: bmwcra_1 goes from 148,770 to 53,877 supervariables, pwtk from 217,918 to 41,531.
- Parallel ND with 200-supervariable AMD leaves predicts 1.019x METIS flops on both.

What the code does today (read before Task 1):
- `make_ordering_backend` / `MetisOrdering::compute_ordering` (`ordering_metis.cpp`): copies xadj/adjncy, then calls `metis_node_nd` with `METIS_SetDefaultOptions` + SEED 0 + NUMBERING 0 under `metis_mutex()`.
- `parallel_nd_order` (`ordering_parallel_nd.cpp`) runs `compress()` and then `bisect()` for each node of a level, through `run_tasks`. METIS calls are serialised. Leaves use `order_leaf`: AMD below `kParallelNdAmdLeafMax` = 200, otherwise `metis_node_nd` with weights. `kParallelNdLevels` = 4 is fixed, so the order does not depend on the thread count.
- `symbolic_analysis.cpp` step 5 (around lines 652-693), when `kHaveMetis && amd_ordered && AutoParallel && n > 8192`:
  - it tries the METIS candidate if `factor_flops(sym0) * expected_refactors >= kMetisFlopsPerRow * n`;
  - it uses `parallel_nd_order` if `num_threads > 1 || deterministic_mode`, otherwise serial METIS on `build_symmetric_adjacency(C)`;
  - it keeps the candidate if its flops are lower;
  - it counts all of this in `phases.ordering`, together with the AMD pass above.
- `select_native_path` (`dispatch_policy.cpp`) runs on the finished `AnalysisKeep`, so it cannot spare analysis work. For SPD + AutoParallel it only diverts bandwidth ≤ 128 to `BandedSPD`. This plan does not change it.
- `bench_replay`'s HSL adapter (`adapters/hsl.cpp:290`) switches SPD recommended cases from `AutoSerial` to `AutoParallel`, which makes this the path the board measures.
- `ordering_matching.cpp:232` (the indefinite matching ordering) also calls `metis_node_nd`. Its options and results must not change in this plan.

SPD cases of the front-tier manifest:
- tuning split: nd24k, oilpan, thermal1, Dubcova3, af_shell7, apache2, thermal2, G3_circuit;
- heldout split: finan512, cfd2, bmwcra_1, pwtk.

**cfd2 and bmwcra_1 are held out.** Profile them freely, but pick every threshold and option on the tuning split. They are validated once, in Task 7.

## Global Constraints

- **Platform:** Windows, MSVC 2022, CMake. cmake and ctest are not on PATH; they live in `"/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/"`.
  - **BUILD:** `nice -n 10 "<that dir>/cmake.exe" --build build --config Release --parallel 4`.
  - **CTEST \<Suite\>:** `"<that dir>/ctest.exe" --test-dir build -C Release -R <Suite> --output-on-failure`.
- Threads are `std::thread` only, never OpenMP. MSVC's OpenMP runtime aborts next to MKL/CHOLMOD.
- **Accuracy gate** (score.py): eta ≤ 1e-10, omega ≤ 1e-8 on every case that is accurate today.
- **Determinism:** in deterministic mode (`deterministic_mode(ctrl)`), the permutation is identical at every `num_threads`. `ParallelNd.DeterministicAcrossThreads` and the `ParallelDeterminism` / `TreeParallel` suites pass unchanged. Any new ND variant must not let the thread count into its result (fixed depth, fixed seeds, results gathered in a fixed order).
- **Indefinite path untouched:** `metis_node_nd` callers that pass no options (`ordering_matching.cpp`, `MetisOrdering`) behave bit for bit as today. Check with a 1T A/B over the indefinite classes in Task 7.
- **Tuning:** only on the `tuning` split. Heldout runs once, in Task 7.
- **Benchmarks:**
  - A/B is interleaved, from one binary, on one machine: `tools/bench/ab_run.py`, with variants that differ only in their `SMF_BENCH_*` environment;
  - results are compared with `tools/bench/ab_compare.py` and `tools/bench/score.py` (10 ms shift);
  - run at below-normal priority, one benchmark at a time (first check that no `bench_replay.exe` is running);
  - ask the user before any run longer than 10 minutes or using more than 1 thread (it is their gaming PC).
- **Dependencies:** switching the linked METIS build (Task 5) is the user's decision. Ask before changing `find_package(METIS)` or the vendored deps.
- **Never** push, commit CoinHSL files, the `.gitignore` change, or `*.log`.
- **Code style:** 2-space indent in `solver/src/` (`ordering_*.cpp` use 4: match the file), 4 in `solver/tests/`. Every new `Control`/`Info` field gets a comment with the evidence for its default. Every new gtest suite gets its own `add_test` in `solver/tests/CMakeLists.txt`.

### Gates (used by Tasks 3-7)

Each gate compares the candidate with the baseline variant (today's behaviour) from the same binary.

- **G-fill:** per SPD case, the kept ordering's `predicted_flops` ≤ 1.03x and `predicted_factor_entries` ≤ 1.03x the baseline's; geomean over the class ≤ 1.01x. Use the record fields `flops` / `factor_entries`, or `bench_ordering`'s columns.
- **G-analyse:** SPD class `analyse` metric (ab_compare `--metric analyse`), shifted geomean ≤ 0.85 at 1T and ≤ 0.70 at 8T. No SPD case gets worse than 1.05x.
- **G-factor:** SPD `--metric factor` shifted geomean ≤ 1.02 at 1T and 8T.
- **G-wall:** SPD `--metric wall` shifted geomean < 1.00 at 1, 8 and 16 threads. No case gets worse than 1.05x.
- **Target, not a gate:** cfd2 analysis ≤ 540 ms at 1T, which is PARDISO's whole analysis.

## Review Focus

1. **METIS concurrency.** If Task 5 removes the mutex, a concurrent METIS call that is not thread-safe corrupts the heap or aborts, sometimes only under load. Tested in Task 5 by `ParallelNd.RawConcurrentMetisIsSafe`: no mutex, 16 threads × 200 calls, results equal to serial, run 20 times under ctest `--repeat until-fail:20`. If it fails even once, stop and report.
2. **Thread count leaking into the order.** Options and leaf scheduling must not depend on `num_threads`. Tested in Task 4: `ParallelNd.DeterministicAcrossThreads` is extended to the new options and to 1/3/8/16 threads, and `Symbolic.SpdOrderSameAtEveryThreadCount` analyses one SPD fixture at 1, 2 and 8 threads with `Control::deterministic` and compares perm.
3. **The indefinite METIS callers.** The new options parameter defaults to "today", and `ordering_matching.cpp` must not pick up SPD options. Tested in Task 3: `MetisOptions.DefaultsMatchSetDefaultOptions` (a null options pointer gives a bitwise-equal perm) plus the existing `MetisOrdering.Matching*` tests.
4. **AMD skip/overlap (Task 6).** Skipping AMD must never pick an ordering with more flops than AMD would have given on a case where AMD wins today (G3_circuit, apache2, thermal2 are the likely ones). Tested in Task 6 by `Symbolic.SpdNdFirstKeepsAmdWinners`, plus a G-fill check over all SPD cases.
5. **Arena reuse in step 5.** The METIS candidate must not alias `region` / `amd_work` while AMD's arrays are live, above all when AMD and ND run concurrently (Task 6). Reviewed against the layout comment at `symbolic_analysis.cpp:419`. ASan is unavailable on MSVC Release, so the test runs the analysis twice on one `Solver` and compares.

---

### Task 1: Measurement — ordering sub-timers and the profile

**Files:**
- Modify: `solver/include/smf/info.hpp` (`AnalysePhaseSeconds`; new `Info` fields)
- Modify: `solver/src/symbolic_analysis.cpp` (step 4/5 timers)
- Modify: `solver/src/ordering_parallel_nd.cpp` (optional stats out-parameter)
- Modify: `solver/include/smf/ordering_parallel_nd.hpp`
- Modify: `solver/benchmarks/common/validation.cpp:146` (JSON), `solver/benchmarks/common/result.hpp`, `solver/benchmarks/bench_profile.cpp` (print)
- Test: `solver/tests/test_parallel_nd.cpp`, the existing analyse-phase test (grep `analyse_phases` in `solver/tests`)

**Interfaces:**
- `AnalysePhaseSeconds` gains `ordering_amd`, `ordering_nd_graph` (adjacency build or compression), `ordering_nd` (METIS / parallel ND call) and `ordering_nd_counts` (`symbolic_counts` of the candidate). The struct comment states that these four sub-split `ordering` and are not part of the partition of `analyse_seconds`.
- `Info::nd_supervariables` (Int, 0 when no ND ran) and `Info::nd_candidate_flops` (double, 0 when not tried). These show compression and the AMD-vs-ND margin per case.
- `struct ParallelNdStats { double compress_s, separators_s, leaves_s, metis_wait_s; Int supervariables; };` passed as an optional last parameter, `ParallelNdStats* stats = nullptr`, to the raw-array `parallel_nd_order`. `metis_wait_s` is the time threads spend blocked on `metis_mutex()`, measured around the `lock_guard`. Expose it through an `ordering_metis.hpp` overload that takes `double* wait_s`.

- [ ] **Step 1: Tests first.**
  - `ParallelNd.StatsAddUp`: on `grid(30,30,30)` at 4 threads, the stats are non-negative, `supervariables` equals what `compress_closed_adjacency` returns, and the perm is unchanged with and without `stats`.
  - `Symbolic.OrderingSubPhasesWithinOrdering`: analyse a 3D Poisson with n > 8192 using `AutoParallel`. Check `ordering_amd + ordering_nd_graph + ordering_nd + ordering_nd_counts <= ordering + 1e-6`, and `ordering_nd > 0` exactly when `nd_candidate_flops > 0`.
- [ ] **Step 2: CTEST ParallelNd, Symbolic.** Expected: FAIL (fields missing).
- [ ] **Step 3: Implement.** Add laps inside the existing `lap()` pattern in `analyse` without moving the `phases.ordering` boundaries. Emit the new fields in `validation.cpp` under `analyse_phases` (`"ordering_amd"`, ...) and print them in `bench_profile` next to `ordering`.
- [ ] **Step 4: BUILD, CTEST ParallelNd, Symbolic.** Expected: PASS.
- [ ] **Step 5: Profile** (ask first: about 10 min at 1/8/16T).
  - `bench_profile --case <id> --ordering auto_parallel --threads T --repeats 3` for cfd2, bmwcra_1, nd24k, pwtk, af_shell7, G3_circuit at T = 1, 8, 16.
  - `bench_profile --sample analyse --case cfd2 --seconds 20 --lines` at 1T.
  - Record:
    - the sub-timers and supervariables;
    - AMD vs ND flops and which ordering was kept;
    - `metis_wait_s` at 8/16T;
    - the hottest METIS functions: coarsening (`Match_SHEM`/`Match_RM`, `CreateCoarseGraph`), refinement (`FM_2WayNodeRefine*`), and `MlevelNodeBisectionMultiple`.
  - Write the table into a new `docs/performance/2026-11-0x-spd-ordering-cost.md` ("Profile" section).
- [ ] **Step 6: Decide the order of Tasks 3-6 from the profile,** and write the decision into the write-up:
  - if `ordering_amd` is under 10% of `ordering` on every case where ND is kept, mark Task 6 "skip";
  - if `metis_wait_s` at 8T is under 20% of `ordering_nd`, Task 5 drops in priority.
- [ ] **Step 7: Commit** `perf(analyse): ordering sub-phase timers and ND stats`.

### Task 2: `bench_ordering` — an ordering-only microbench

**Files:**
- Create: `solver/benchmarks/bench_ordering.cpp`
- Modify: `solver/benchmarks/CMakeLists.txt` (`smf_benchmark(bench_ordering)`, next to `bench_matching_trial`)

**Interfaces:**
- `bench_ordering --manifest benchmarks/manifests/fronttier.json --class spd [--split tuning] [--cases a,b] --variants v1,v2,... [--threads T] [--repeats R]`.
- It loads each matrix the way `bench_replay` does (`.smfbin` cache), builds the arena adjacency with `lower_csc_adjacency` (as in analyse), and for each variant prints best-of-R ms, `predicted_flops`, nnz(L) and the ratios to variant 1. It computes flops by passing the perm through `Solver::analyse` with `OrderingMethod::UserSupplied` (as `predicted_flops()` in `test_parallel_nd.cpp`), outside the timed region.
- Variants:
  - `amd`;
  - `metis` (today's `MetisOrdering`);
  - `metis:<opts>` (Task 3 options string, e.g. `niter=3,ctype=rm,no2hop=1`);
  - `cmetis[:<opts>]` (compressed graph + vwgt into `metis_node_nd`, expanded);
  - `pnd[:levels=L,leaf=K,<opts>]` (`parallel_nd_order`).

- [ ] **Step 1:** Write it. There is no gtest. Its self-check is that every perm is a permutation of 0..n-1; it aborts the case otherwise.
- [ ] **Step 2: BUILD.** Smoke: `bench_ordering --cases thermal1 --variants amd,metis,pnd --repeats 1` prints three rows, with ratio 1.000 on `metis`.
- [ ] **Step 3: Baseline:** run `--class spd --variants amd,metis,cmetis,pnd --repeats 3` at 1T over all 12 SPD cases. This is profiling, so heldout is allowed. Add the table to the write-up.
- [ ] **Step 4: Commit** `bench(ordering): ordering-only microbench`.

### Task 3: Per-call METIS options

**Files:**
- Modify: `solver/include/smf/ordering_metis.hpp`, `solver/src/ordering_metis.cpp`
- Modify: `solver/include/smf/ordering_parallel_nd.hpp` (`ParallelNdOptions::metis`)
- Modify: `solver/include/smf/control.hpp` (`Control::nd_metis_fast`), `solver/src/symbolic_analysis.cpp` (step 5 passes it)
- Modify: `solver/benchmarks/bench_replay.cpp` (env knob next to `SMF_BENCH_AMALGAMATION`), `bench_ordering.cpp` (options parser)
- Test: `solver/tests/test_ordering_metis.cpp`

**Interfaces:**
- `struct MetisOptions { int ctype = -1, niter = -1, nseps = -1, no2hop = -1, compress = -1, ccorder = -1, pfactor = -1, ufactor = -1; };` where -1 keeps METIS's default.
- `bool metis_node_nd(Int n, Int* xadj, Int* adjncy, Int* vwgt, Int* perm, Int* iperm, const MetisOptions* opt = nullptr);` and the same trailing parameter on `metis_vertex_separator`.
- `metis_options(idx_t*, const MetisOptions*)` applies the non-negative fields after `METIS_SetDefaultOptions` and SEED/NUMBERING. Confirm each `METIS_OPTION_*` name against the linked `metis.h` (5.2.1); skip any it lacks, with a comment.
- `Control::nd_metis_fast` (bool, default false until Task 7). When true, step 5's ND candidate (serial or parallel) uses `kSpdNdFastMetis` (constexpr `MetisOptions` in `ordering_metis.hpp`, chosen in Step 5).
- Env knob `SMF_BENCH_ND_FAST = 0 | 1` sets `ctrl.nd_metis_fast`.

- [ ] **Step 1: Tests.**
  - `MetisOptions.DefaultsMatchSetDefaultOptions`: on `grid` 20³ and bmwcra_1 (when present), `opt = nullptr` and `MetisOptions{}` give a perm bitwise equal to today's.
  - `MetisOptions.FastOptionsValidAndClose`: `kSpdNdFastMetis` on 3D Poisson 30³ gives a permutation whose predicted flops are ≤ 1.05x of the default's. This is a loose unit bound; G-fill is the real gate.
  - The existing `MetisOrdering.*` tests pass unchanged.
- [ ] **Step 2: CTEST OrderingMETIS.** Expected: the new tests FAIL to compile.
- [ ] **Step 3: Implement** the parameter and options plumbing. Leave `kSpdNdFastMetis` as all -1 for now.
- [ ] **Step 4: BUILD, CTEST OrderingMETIS, ParallelNd.** Expected: PASS.
- [ ] **Step 5: Sweep on tuning only.** Run `bench_ordering --split tuning --class spd` at 1T, varying one knob at a time from the default:
  - `niter` = 10 (default), 5, 3, 2;
  - `ctype` = shem (default), rm;
  - `no2hop` = 0, 1;
  - `nseps` = 1 (default), and 2 to check whether better separators pay for themselves;
  - `compress` / `ccorder` on nd24k only;
  - `pfactor` = 0, 50, 100 on nd24k (dense rows).

  Then combine the winners. Choose the setting with the lowest ordering-time geomean that holds per-case flops and nnz(L) ≤ 1.02x (leaving slack under G-fill's 1.03). Set `kSpdNdFastMetis` to it, with a comment giving the sweep numbers.
- [ ] **Step 6: A/B** (ask first: 1T and 8T, tuning SPD, 3 rounds):
  ```
  python tools/bench/ab_run.py --manifest benchmarks/manifests/fronttier.json --split tuning --classes spd \
    --rounds 3 --threads 1 --output benchmarks/results/nd_fast_ab_1t \
    --variant base=SMF_BENCH_ND_FAST=0 --variant fast=SMF_BENCH_ND_FAST=1
  python tools/bench/ab_compare.py --a benchmarks/results/nd_fast_ab_1t/base --b benchmarks/results/nd_fast_ab_1t/fast \
    --manifest benchmarks/manifests/fronttier.json --metric analyse
  ```
  Repeat with `--metric factor` and `--metric wall`, then the same at `--threads 8`. Record G-fill, G-analyse, G-factor and G-wall in the write-up. Keep the default false; Task 7 flips it.
- [ ] **Step 7: Commit** `perf(ordering): per-call METIS options; fast SPD ND options behind Control::nd_metis_fast`.

### Task 4: Compression on the 1-thread path (`parallel_nd_order` at every thread count)

**Files:**
- Modify: `solver/include/smf/control.hpp` (`Control::nd_strategy`), `solver/src/symbolic_analysis.cpp` (step 5 branch at about line 665)
- Modify: `solver/src/ordering_parallel_nd.cpp` (pass `ParallelNdOptions::metis` through `bisect` and `order_leaf`)
- Modify: `solver/benchmarks/bench_replay.cpp` (`SMF_BENCH_ND_STRATEGY`)
- Test: `solver/tests/test_parallel_nd.cpp`, `solver/tests/test_symbolic.cpp`

**Interfaces:**
- `enum class NdStrategy { Auto, SerialMetis, ParallelNd };` and `Control::nd_strategy = NdStrategy::Auto`.
  - `Auto` is today's rule (serial METIS at 1T unless deterministic) until Task 7.
  - `SerialMetis` forces `MetisOrdering`.
  - `ParallelNd` forces `parallel_nd_order` at every thread count, `num_threads` = 1 included.
- `ParallelNdOptions` gains `MetisOptions metis;`.
- Env knob `SMF_BENCH_ND_STRATEGY = auto | serial | parallel`.
- Reuse the arena adjacency (`xadj`, `adj`) for the serial path too. `SerialMetis` copies it into a `std::vector` (METIS wants mutable arrays) instead of calling `build_symmetric_adjacency(C)` again. The perm is unchanged because it is the same graph; assert that in the test.

- [ ] **Step 1: Tests.**
  - `Symbolic.SerialMetisFromArenaSameOrder`: on 3D Poisson 25³ at 1T, `SerialMetis` gives the same perm as `MetisOrdering::compute_ordering` on `build_symmetric_adjacency`.
  - `Symbolic.SpdOrderSameAtEveryThreadCount`: `ParallelNd` plus `deterministic`, at 1/2/8 threads, gives an identical `perm`.
  - `ParallelNd.DeterministicAcrossThreads` is extended to 1/3/8/16 threads with `ParallelNdOptions{.metis = kSpdNdFastMetis}`.
- [ ] **Step 2: CTEST Symbolic, ParallelNd.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, CTEST Symbolic, ParallelNd, ParallelDeterminism, TreeParallel.** Expected: PASS.
- [ ] **Step 5: Tune** on tuning SPD with `bench_ordering` at 1T and 8T. Vary `pnd:levels=` 2, 4, 6 and `leaf=` 100, 200, 400, with and without the Task 3 options. Rules:
  - levels stay fixed, never derived from `num_threads`;
  - the per-case G-fill bound against today's kept ordering applies;
  - if `pnd` at 1T is not faster than serial `metis` on most tuning cases, record that and keep `Auto` = serial at 1T.
- [ ] **Step 6: A/B** (ask first) at 1T and 8T over tuning SPD:
  ```
  ab_run.py ... --variant base=SMF_BENCH_ND_STRATEGY=auto \
                --variant pnd=SMF_BENCH_ND_STRATEGY=parallel \
                --variant pndfast=SMF_BENCH_ND_STRATEGY=parallel,SMF_BENCH_ND_FAST=1
  ```
  Use ab_compare with `analyse`, `factor` and `wall`. Gates: G-fill, G-analyse, G-factor.
- [ ] **Step 7: Commit** `perf(analyse): SPD ND from the arena adjacency; Control::nd_strategy`.

### Task 5: Thread-safe METIS (truly concurrent separators and leaves)

**Decision point:** this changes a dependency. Before Step 3, show the user the Step 1 result and ask whether to switch the linked METIS.

**Files:**
- Modify: `solver/CMakeLists.txt` (option `SMF_METIS_THREADSAFE`, default OFF; when ON, define `SMF_METIS_THREADSAFE=1`)
- Modify: `solver/src/ordering_metis.cpp` (`metis_mutex()` lock becomes a no-op under the define)
- Modify: `solver/include/smf/ordering_parallel_nd.hpp` (doc comment: METIS calls are concurrent when thread-safe)
- Test: `solver/tests/test_parallel_nd.cpp`

**Interfaces:**
- `bool metis_is_thread_safe() noexcept;` (constexpr-backed by the define).
- Build METIS 5.2.1 + GKlib from source (KarypisLab tags `v5.2.1` / matching GKlib) with thread-local GKlib state. GKlib's `gk_arch.h` / `memory.c` declare `gkmcore` and `gk_cur_jbufs` with `GK_THREADLOCAL`; check that the MSVC build expands it to `__declspec(thread)` and not to nothing. That expansion is the likely difference from the conda-forge DLL that aborts. Install to `build-deps/metis-ts/` and point `METIS_ROOT` at it.

- [ ] **Step 1: Investigate before you build anything new.** Read the linked METIS's GKlib headers and check how `GK_THREADLOCAL` expands under the conda-forge build flags. Record whether a from-source build with thread-local storage is enough, or whether another global (`gk_jbuf`, `gk_sigtrap`) needs a patch. Report to the user; stop if a patch to METIS itself would be needed.
- [ ] **Step 2: Test** `ParallelNd.RawConcurrentMetisIsSafe` (compiled only under `SMF_METIS_THREADSAFE`). It calls `METIS_NodeND` and `METIS_ComputeVertexSeparator` directly, with no mutex, from 16 threads × 200 calls over 8 grids, and the results equal the serial ones. Also `ParallelNd.MetisWaitZeroWhenThreadSafe`: under the define, `ParallelNdStats::metis_wait_s == 0`.
- [ ] **Step 3:** After the user approves, configure a second build dir `build-ts` with `-DSMF_METIS_THREADSAFE=ON -DMETIS_ROOT=build-deps/metis-ts`. BUILD it, then run `CTEST ParallelNd` with `--repeat until-fail:20`. **If it fails once, stop and report.**
- [ ] **Step 4: Make the levels concurrent.** In `parallel_nd_order`, the split of one level and the leaves already run on `run_tasks`. With METIS concurrent, start a child's split as soon as its parent finishes instead of waiting for the level barrier. Use a task queue over `nodes`, with the tree shape fixed by the graph alone. Keep the emitted order a function of the tree only, and keep `DeterministicAcrossThreads` green.
- [ ] **Step 5: Measure** with `bench_ordering --variants pnd,pnd:<fast>` at 1/8/16T in `build` vs `build-ts`, on tuning SPD. Expect ordering at 8T ≤ 0.5x of 1T on af_shell7 and nd24k. The top separator stays serial: report its share from `ParallelNdStats::separators_s` per depth.
- [ ] **Step 6: Commit** `perf(ordering): concurrent METIS when built thread-safe (SMF_METIS_THREADSAFE)`. Commit the CMake option and code only, never the built deps.

### Task 6 (conditional on Task 1 Step 6): Overlap or skip the AMD pre-pass

Skip this task if Task 1 showed `ordering_amd` < 10% of `ordering` wherever ND is kept.

**Files:**
- Modify: `solver/src/symbolic_analysis.cpp` (step 4/5), `solver/include/smf/control.hpp` (`Control::spd_nd_first`)
- Modify: `solver/benchmarks/bench_replay.cpp` (`SMF_BENCH_ND_FIRST`)
- Test: the Symbolic suite

**Interfaces:**
- `Control::spd_nd_first` (bool, default false). When true and `num_threads > 1`, SPD + AutoParallel + n > 8192 starts the ND candidate on a worker `std::thread` while AMD runs on the caller. Both run on separate arena regions or their own vectors; see Review Focus 5. The keep rule is unchanged: ND is kept only if AMD's flops pass `kMetisFlopsPerRow` and ND's flops are lower. At 1T the flag does nothing unless Task 1 found a cheap and exact pre-predictor.
- Env knob `SMF_BENCH_ND_FIRST = 0 | 1`.

- [ ] **Step 1: Tests.**
  - `Symbolic.SpdNdFirstSameChoice`: on two fixtures, a 3D Poisson (ND wins) and a 2D 5-point grid with n ≈ 10k (check which wins), `spd_nd_first` at 4 threads gives the same `ordering_used` and perm as the flag off.
  - `Symbolic.SpdNdFirstKeepsAmdWinners`: on a graph where AMD wins (a long thin 2D strip, or G3_circuit when present), `ordering_used == AMD`.
  - `Symbolic.AnalyseTwiceSameSolver`: two analyses on one `Solver` with the flag on give equal keeps (Review Focus 5).
- [ ] **Step 2: CTEST Symbolic.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: BUILD, CTEST Symbolic, ParallelDeterminism.** Expected: PASS.
- [ ] **Step 5: A/B** (ask first) at 8T and 16T on tuning SPD, `SMF_BENCH_ND_FIRST=0` vs `1`, ab_compare `analyse` and `wall`. Gate: G-analyse at 8T, plus no case worse than 1.05x.
- [ ] **Step 6: Commit** `perf(analyse): SPD ND candidate overlaps AMD at num_threads > 1`.

### Task 7: Policy, heldout validation and write-up

**Files:**
- Modify: `solver/include/smf/control.hpp` (the defaults of `nd_metis_fast`, `nd_strategy`'s `Auto` rule, `spd_nd_first`, each with its evidence comment)
- Modify: `solver/src/symbolic_analysis.cpp` (the `Auto` rule)
- Modify: `docs/performance/2026-11-0x-spd-ordering-cost.md` (results), `docs/roadmap.md` (one line in the analyse gap entry)

- [ ] **Step 1:** Set the defaults to the variants that passed G-fill, G-analyse and G-factor on tuning. Update the comments (`ordering_parallel_nd.hpp` header, `symbolic_analysis.cpp` step 5, `ordering_metis.cpp` factory) to the new behaviour.
- [ ] **Step 2: BUILD, full CTEST.** Expected: PASS.
- [ ] **Step 3: Board** (ask first; about 30-40 min in total):
  - all 12 SPD cases, both splits, at 1, 8 and 16 threads, 3 interleaved rounds: old defaults (all knobs set to today via env) vs new defaults;
  - plus the indefinite classes (`hard_indef,large_kkt,small_kkt`) at 1T, tuning split, 1 round, as an unchanged check. ab_compare wall within ±2% (noise), and `factor_entries` identical per case.
- [ ] **Step 4: Score** with `tools/bench/score.py --records <new> --manifest benchmarks/manifests/fronttier.json --class spd` against the stored rival board, and report cfd2, bmwcra_1 and nd24k ratios vs PARDISO/SSIDS at each thread count. Gates: G-fill, G-analyse, G-factor, G-wall on heldout, with no heldout gate worse than tuning by more than 10%, and the accuracy gate on all cases. Report whether the cfd2 ≤ 540 ms target was met.
- [ ] **Step 5: Write-up:** profile, sweep tables, A/B tables, what was ruled out and why, and the open items. One example of an open item: the top separator stays serial, so a parallel multilevel bisection (ParMETIS/mt-metis-style) would be the next lever.
- [ ] **Step 6: Commit** `perf(analyse): cheaper SPD nested dissection by default` and `docs(performance): SPD ordering cost`.
