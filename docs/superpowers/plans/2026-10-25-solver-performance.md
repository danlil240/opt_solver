# SMF Performance Leadership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Earn a measured native SMF advantage on repeated KKT systems, then qualify broader leadership against applicable HSL and modern competitors.

**Architecture:** Keep the C++20 symbolic infrastructure and SPD path; repair indefinite numerics, separate immutable analysis from reusable numeric/solve workspaces, and introduce blocked arithmetic and costed parallelism. A robust FP64 engine remains the fallback and reference for optional fast paths.

**Tech Stack:** C++20, CMake, BLAS/LAPACK, OpenMP, AMD/METIS, GoogleTest, Python 3 for benchmark orchestration/reporting, real IPOPT and separately installed comparator libraries.

**Spec:** [2026-10-25-solver-performance-design.md](../specs/2026-10-25-solver-performance-design.md). Read it first; it defines all numerical, benchmarking, claim and resource gates.

**Status:** Planning only, based on revision `b43b703`. No steps below have been implemented or tested during plan creation. Future command names/files are explicitly proposed. No existing benchmark number is a new measurement.

## Global Constraints

- Keep C++20, lower-triangular CSC, double precision as the reference, and existing useful symbolic infrastructure.
- For nonsingular success, every answer is finite and solves the original unscaled, unregularized system to normwise backward error `eta_inf <= 1e-10` and componentwise backward error `omega <= 1e-8`.
- No silent regularization or falsely successful singular solve.
- Scaling, refinement, retries, fallback and transfers count toward the relevant solve/sequence time.
- Missing required HSL results block an HSL leadership claim, not independent development.
- Native SMF and whole-system external backend dispatch are separate result identities. Never label `libsmf_ma97` as genuine HSL MA97.
- Preserve historical plans/logs. Correct old benchmark reports with dated notices and links to generated evidence, rather than overwriting history.
- Keep `.live-agents` current before build/test/install; only one exclusive build/test/install owner. Read-only audits and disjoint edits may run in parallel. Create isolated worktrees at implementation time where needed.
- Separate optimized benchmark and sanitizer builds. Do not benchmark with fast-math or relaxed numerical tolerances. Use Linux as the first supported benchmark platform; Windows ABI support requires explicit qualification.
- No purchases, rental, publication or redistribution of licensed baselines is included in execution authorization.

## Review Focus

1. A nonsingular matrix needs a 2x2 pivot, a numeric swap, or a pivot delayed across fronts: accurate solve and inertia, no false singularity (Tasks 2-3).
2. Repeated values arrive unsorted/duplicated in the original input ordering, or sparsity changes: correct remapping or explicit reanalysis, never stale factors (Task 4).
3. Extreme scaling, tiny determinants and ambiguous rank: finite scaled arithmetic, explicit status, no unconditional success (Tasks 2, 5).
4. Multiple RHS, released analysis handles and concurrent independent solves: owned factor metadata and caller-specific scratch (Tasks 2, 4, 9).
5. Missing baseline, inaccurate fast result, overflow/OOM, or differing BLAS threads: fail qualification and retain the reason, never inflate speedups (Tasks 1, 4, 6, 8).

## Dependency order and ownership

`Task 1 -> Tasks 2 and 6 in parallel -> Task 3 -> Task 4 -> Tasks 5 and 7 -> Tasks 8 and 9 -> Task 10 -> Task 11`

Task 6 can continue independently once Task 1's result contract is frozen. Task 9's ABI tests can begin early, but final qualification needs the corrected core. Tasks 5 and 7 share factor internals: merge interface changes first and serialize overlapping edits. Early profiling is useful; performance claims require G1.

Suggested owners: numerical specialist (2/3/5/7), performance engineer (4/8/10), benchmark/integration engineer (1/6/9). A separate reviewer examines every numerical representation change. Parallel agents receive disjoint file ownership and one numerical-contract version; do not generate several incompatible factor formats.

## Task 1: Make wrong or missing answers impossible to call wins

**Files:** create `solver/benchmarks/common/{result.hpp,validation.hpp,validation.cpp}`, `solver/tests/test_benchmark_validation.cpp`, `tools/bench/{run.py,report.py}`, `tools/bench/test_report.py`; modify `solver/benchmarks/{bench_compare.cpp,bench_kkt_fixture_compare.cpp,bench_suite_sparse_matrix_market.cpp,CMakeLists.txt}`, `solver/{CMakeLists.txt,tests/CMakeLists.txt}`, `docs/benchmark_results.md`, `docs/benchmark_report.md`.

**Interfaces:** namespace `smf::bench`; `Outcome {Success, Inaccurate, Singular, InvalidInput, Unavailable, Timeout, OutOfMemory, Failed}`; `Validation {bool finite; double eta_inf, omega, relative_residual;}`; `Validation validate(const CscLower&, std::span<const double> b, std::span<const double> x, int nrhs)` with buffers of length `n*nrhs`. `RunRecord` contains schema version, matrix/sequence IDs and hashes, solver identity/version, configuration, outcome/reason, phase times, wall time, peak RSS, validation per RHS/update, inertia and raw trial ID. Serialize one record per attempt to JSONL; never drop attempts.

- [ ] Add `BenchmarkValidation` tests: exact manufactured solution passes; corrupted solution fails; NaN/Inf fails; zero RHS follows the spec's denominator convention; a bad second RHS fails; a success return with excessive backward error becomes `Inaccurate`.
- [ ] Add Python reporting tests: `Unavailable` cannot win; timeout/OOM remain in denominators; no jointly solved cases produces "insufficient coverage", not a geometric mean; an SMF failure blocks qualification; family weights and 95% interval resampling preserve sequence grouping.
- [ ] Run tests and record expected failures before implementing validation/report logic. Share validation code across existing harnesses; replace permissive `res>=0` checks and unconditional zero exit codes. CLI returns nonzero for requested required-baseline absence, input failure, or qualification failure.
- [ ] Separate warmups from measurements; support randomized solver order and isolated subprocesses. Implement cold, changed-value sequence and existing-factor RHS modes with the spec's timing boundaries. Validate outside timed regions but include solver-internal refinement/retry time. Batch tiny cases; retain at least 10 measured trials.
- [ ] Inventory toolchain, BLAS integer/thread ABI, actual comparator library symbols/paths/versions, available hardware and fixtures. Detect OpenBLAS before emitting compile definitions; make `SMF_USE_MKL` actually select/verify the linked backend. Add `SMF_REQUIRE_BENCHMARK_BASELINES` (default OFF); release comparison presets set it ON with a required solver list.
- [ ] Use the tested Matrix Market reader, reject inconsistent formats/counts, validate all RHS, and collect isolated-process resident memory. Add dated historical-result notices and remove unsupported explanations from current guidance.
- [ ] Verify `ctest --test-dir build --output-on-failure -R BenchmarkValidation` and `python -m unittest discover -s tools/bench -p 'test_*.py'`; exercise injected solver failures and missing-library errors. Commit this task with the raw failing/passing harness evidence. Exit: G0 harness semantics, not solver correctness.

## Task 2: Repair pivot representation and solve semantics

**Files:** modify `solver/src/{pivoting.cpp,dense_indef.cpp,factor_indef.cpp,solve_forward.cpp,solve_backward.cpp,solve_diag.cpp,solve.cpp,solver.cpp}`, `solver/include/smf/{factor_posdef.hpp,pivoting.hpp}`; create `solver/tests/{test_indef_adversarial.cpp,test_factor_identity.cpp}`; extend `test_pivoting.cpp`, `test_inertia_dense.cpp`, `tests/CMakeLists.txt`.

**Interfaces:** retain public `Solver::analyse/factor/solve` initially. In `FactorKeep`, store D exclusively in `DiagSolveEntry` blocks and make every 2x2 pivot's internal L block identity; add explicit numerical permutation metadata. Define one total `perm[new]=old` interpretation for completed factors and every solve path. Solve failures must propagate into return status and `Info`, including `factor_solve`.

- [ ] Add `IndefAdversarial.TwoByTwoExchange`: force one full front with `A=[[0,1],[1,0]]`, `b=[1,2]`; assert `x=[2,1]` within `1e-12`, inertia `(1,1,0)`, and the common backward-error gates. Cover tiny, generic, and multiple-RHS kernels.
- [ ] Add `IndefAdversarial.SecondaryPivotZeroDiagonal` with the spec's full 3x3 front and `u=.01`; assert finite factors/solution and no zero accepted 1x1 divisor. Add a non-adjacent numeric-swap case and dense-helper equivalents. Confirm failures before the fix; if dispatch avoids the intended path, test the front kernel directly as well as the public API.
- [ ] Add `FactorIdentity.NumericPermutationAndBlockD`: reconstruct `P^T A P` and `L D L^T`; require normalized reconstruction error `<=1e-12` on small well-scaled fixtures. Repeat after a front has already accepted earlier pivots; check permutation bijection and consistency of row labels.
- [ ] Fix pivot tests, record swaps, separate D from L, and use scaled 2x2 solve/inertia arithmetic without raw overflow-prone determinant tests. Add fixtures scaled by `1e-100` and `1e100`, with consistent rank/finite-status policy.
- [ ] Test singular and invalid-input failure propagation, unknown/unsupported solve jobs, dimension/RHS/leading-dimension checks, and a returned factor used after its originating analysis object is destroyed. Preserve SPD behavior.
- [ ] Register `IndefAdversarial` and `FactorIdentity`; run them plus `Pivoting|LDLTDense|InertiaDense|Solve|IndefLarger|Singular|OcpKktRegression`, then the full suite and ASan/UBSan build. Obtain independent numerical review and commit only after the intended branches are exercised.

## Task 3: Implement real delayed pivots and full-front stability

**Files:** create `solver/include/smf/numeric_front.hpp`, `solver/src/numeric_front.cpp`, `solver/tests/test_delayed_pivots.cpp`; modify `factor_indef.cpp`, `pivoting.cpp`, `frontal_matrix.cpp`, `include/smf/{factor_posdef.hpp,frontal_matrix.hpp,info.hpp}`, solver/test CMake lists.

**Interfaces:** `NumericFront` owns ordered global row IDs, an eligible-column count and active dense storage; `FrontResult` owns accepted pivot row IDs, L panel, D blocks, and a contribution with explicit row IDs and delayed-column IDs. Expose `FrontResult factor_front_reference(NumericFront&, const Control&)`. Symbolic `FrontalInfo` remains a prediction, not the realized numeric layout. Completed solve metadata derives from realized factor fronts.

- [ ] Add `DelayedPivots.CrossFrontNonsingular`: construct a tiny fixed front tree where a zero local diagonal must meet a nonzero off-diagonal in its parent; assert successful nonsingular factorization, positive delay count, zero nullity and valid residual/inertia. Test multiple levels, multiple children and a genuinely singular root.
- [ ] Add `DelayedPivots.ExtensionRowGrowth`: a tiny pivot with a large extension coupling must fail the stability check or be replaced by an acceptable block. Verify full-column bounds, not only the fully summed square block.
- [ ] Replace skip-and-count-as-zero with active row/column compaction, explicit delayed data, parent front expansion and checked capacity growth. Assemble each original/contribution entry exactly once; preserve numeric labels through permutations.
- [ ] Share reference frontal arithmetic between serial and parallel drivers. Differentiate delayed, singular, allocation and input errors; publish actual accepted counts/inertia only after valid completion.
- [ ] Run `ctest --test-dir build --output-on-failure -R 'DelayedPivots|IndefAdversarial|FactorIdentity|ParallelFactor'`, dense-oracle differential cases and the full sanitizer suite. Compare 1/2/4-thread numerical results; deterministic mode gets its own explicit bitwise scope. Commit after numerical review. Exit: core G1 correctness for tested reference classes.

## Task 4: Make repeated numeric updates efficient and safe

**Files:** create `solver/include/smf/workspace.hpp`, `solver/src/workspace.cpp`, `solver/tests/test_numeric_workspace.cpp`; modify `include/smf/{solver.hpp,analysis.hpp,factor_posdef.hpp,types.hpp}`, `src/{symbolic_analysis.cpp,factor_indef.cpp,factor_posdef.cpp,solve.cpp,solve_forward.cpp,solve_backward.cpp}`, `tests/test_repeated_factor.cpp`, build lists.

**Interfaces:** `NumericWorkspace` owns reusable value buffers, scratch and contribution/factor capacity; `SolveWorkspace` owns per-call scratch. Add `Solver::factor(const AnalysisKeep&, const CscLower& updated, const Control&, Info&, FactorKeep&, NumericWorkspace&) const` and `Solver::solve(const FactorKeep&, const Control&, Info&, double* b, int n, int nrhs, SolveWorkspace&, SolveJob=SolveJob::Full) const`. Preserve old overloads as convenience wrappers. Validate that `updated` has the original analyzed pattern; original-entry maps handle duplicate/unsorted values. A future unchecked values-only overload is not needed for this milestone.

- [ ] Add `NumericWorkspace.ChangedValuesSamePattern`: replay at least 100 value updates without editing analysis internals; validate each against the new matrix, including inertia changes and duplicate/unsorted input. `PatternChanged` must reject or explicitly reanalyze; it must never reuse stale symbolic maps.
- [ ] Add `NumericWorkspace.ConcurrentSolve`: independent workspaces solve against the same immutable factors without races. Test independent factorization instances; do not concurrently mutate the same factor/workspace. Remove reliance on mutable scratch in a const factor handle.
- [ ] Promote factor/storage offsets to 64-bit; keep external index ABIs explicit. Add checked multiplication/addition helpers and `StorageOverflow` tests using metadata-sized cases, without allocating multi-gigabyte buffers. Failure is explicit and leaves no valid-looking partial factor.
- [ ] Precompute original-to-canonical-to-front maps and stable traversal metadata. Reuse value buffers, per-worker arenas and pooled contributions; release consumed contributions to the pool rather than retaining all capacities through the traversal. Reserve predictable capacity while allowing measured growth for pivot changes.
- [ ] Assert no per-front heap allocation in warmed stable-pivot fixtures, and report rather than conceal growth when pivot structure changes. Measure full changed-value sequences against the corrected reference. Require unchanged accuracy and at least a 10% median sequence improvement on the targeted overhead-bound training subset, with no material validation regression, before enabling the new path by default.
- [ ] Run `NumericWorkspace|RepeatedFactor|OcpKktRegression|FactorIdentity|DelayedPivots`, full CTest and appropriate race checks. Commit ownership/ABI changes with migration notes.

## Task 5: Integrate scaling, refinement and symmetric matching

**Files:** create `solver/include/smf/refinement.hpp`, `solver/src/refinement.cpp`, `solver/tests/test_refinement.cpp`; modify `src/{scaling.cpp,scaling_matching.cpp,solver.cpp,solve.cpp}`, `include/smf/{control.hpp,info.hpp,scaling_matching.hpp,workspace.hpp}`, scaling tests/build lists.

**Interfaces:** `RefinementResult {ErrorCode status; int iterations; double eta_inf; double omega;}` and `refine(const CscLower& original, const FactorKeep&, const Control&, std::span<const double> b, std::span<double> x, int nrhs, SolveWorkspace&)`. Add `Control::refinement_max_iters=5`; stop on the spec's accuracy target or stagnation and report failure if qualification is unmet. Record positive diagonal scaling in completed factors; direct and convenience APIs use the same scaling contract.

- [ ] Add `Refinement.OriginalSystemAfterScaling`, `Refinement.StagnationIsNotSuccess`, `Refinement.EveryRhs`, and repeated-value fixtures whose scaling changes substantially. Check the common error gates against original A/b.
- [ ] Apply equilibration consistently in reusable factorization, RHS transformation and solution recovery. Validate scale finiteness/positivity and avoid double scaling across wrapper APIs. Use robust residual accumulation and record refinement work/time.
- [ ] Implement or integrate a permitted symmetric matching/ordering method after documenting its permutation/scaling contract. Keep matching setup, fill inflation and delay reduction visible. Do not import proprietary HSL implementation code. Unsupported matching must remain an explicit error until the implementation is qualified.
- [ ] Add `Matching.SymmetryAndInertiaPreserved`, `Matching.StructurallySingular`, and a difficult scaled KKT sequence. Test mapping composition with numeric swaps and delayed pivots.
- [ ] Run `Refinement|ScalingEquilib|Matching|IndefAdversarial|NumericWorkspace` and the strict benchmark. Enable matching by policy only if its full-sequence benefit survives held-out validation. Commit correctness separately from tuned defaults.

## Task 6: Establish a real competitor and workload laboratory

**Files:** create `solver/benchmarks/adapters/{adapter.hpp,hsl.cpp,spral.cpp,mumps.cpp,pardiso.cpp,cholmod.cpp}`, `solver/benchmarks/bench_replay.cpp`, `benchmarks/manifests/{smoke.json,kkt.json,symmetric.json}`, `tools/bench/capture_manifest.py`, `solver/tests/test_benchmark_adapters.cpp`; extend benchmark CMake and `tools/bench/{run.py,report.py}`.

**Interfaces:** `SolverAdapter` provides `analyse(const CscLower&)`, `factor(std::span<const double>)`, `solve(std::span<double>, int nrhs)`, `identity()`, and structured status/statistics; all calls use the Task 1 result contract. Each adapter declares supported classes, ordering/scaling controls and integer ABI. Runtime identity includes actual library path/version; symbol aliases alone are insufficient.

- [ ] Test every installed adapter on identity SPD, the 2x2 exchange matrix where supported, changed-value replay, multi-RHS, singularity and missing-library handling. Compile against actual headers; check integer widths and symbol signatures. Required-baseline mode fails closed when a library is absent.
- [ ] Implement genuine HSL MA27/57/86/87/97 adapters for applicable classes; add SSIDS, MUMPS, Intel oneMKL PARDISO and separately identified Panua PARDISO when available. Add MA77 only with an explicitly labeled resource/IO track. Use Eigen as an oracle/small-case reference, not the definition of state of the art.
- [ ] Freeze the design's corpus sizes, hashes, families and holdouts. Use existing KKT capture hooks as input, preserving numeric update IDs, RHS batches and failed/retried steps. Avoid leakage from the same originating application into tuning and holdout sets.
- [ ] Run common-configuration diagnostic and recommended-configuration headline tracks with equal tuning budgets; collect actual threads, resident memory, numerical quality and phase distributions. Add allocation/front-size/panel/assembly/solve profiles to locate the dominant cost. Reconcile phase totals with external wall time.
- [ ] Verify proposed CLI: `python tools/bench/run.py --manifest benchmarks/manifests/smoke.json --require-baselines --trials 10 --output build/bench-results/smoke`; `python tools/bench/report.py --input build/bench-results/smoke --gate G0`. Exact CLI is implemented here, not available in the audited revision.
- [ ] Commit adapter/corpus/reporting code and small public fixtures. Store large data/results by manifest and artifact location. Exit: reproducible baseline with unavailable entries explicitly listed; no leadership claim yet.

## Task 7: Replace scalar indefinite updates with blocked arithmetic

**Files:** create `solver/include/smf/blocked_indef.hpp`, `solver/src/blocked_indef.cpp`, `solver/tests/test_blocked_indef.cpp`, `solver/benchmarks/bench_front_kernel.cpp`; modify `factor_indef.cpp`, `dense_kernel_lapack.cpp`, `include/smf/{dense_kernel.hpp,control.hpp}`, build lists.

**Interfaces:** `FrontResult factor_front_blocked(NumericFront&, const Control&, NumericWorkspace&)` returns the same semantic result as Task 3. `Control::indef_panel_size` is a tunable positive integer; use 32 as an initial development value, not an asserted optimum. Reference mode remains selectable for diagnosis.

- [ ] Add differential tests against the reference for panel-boundary 2x2 pivots, swaps after accepted pivots, delays crossing panels, rectangular extension blocks and extreme scaling. Assert reconstructed factor identity, inertia and external accuracy, not identical floating-point factors.
- [ ] Factor panels with the documented threshold policy; form `W=L21*D`, then apply tiled `A22 -= W*L21^T` with suitable GEMM/symmetric block updates. Do not use positive-definite SYRK formulas on indefinite D without a valid transformation. Avoid materializing matrix inverses.
- [ ] Retain tiny-front kernels below measured crossover. Measure panel, update, assembly and packing times across recorded real front shapes; check end-to-end sequence time as well as kernel GFLOP/s.
- [ ] Sweep panel sizes 16/32/64/128 on training data only. Default enablement requires G1, at least 1.5x factor-phase improvement on the declared update-dominated subset, and at least 1.1x full-sequence improvement there; report other cohorts and retain reference routing for regressions.
- [ ] Run `BlockedIndef|DelayedPivots|FactorIdentity|IndefAdversarial`, full CTest and the held-out replay gate. Commit the numerical kernel separately from tuned dispatch defaults.

## Task 8: Make multicore speedup survive memory and root fronts

**Files:** modify `src/parallel/{task_tree.cpp,blas_thread_guard.cpp,determinism.cpp}`, `src/{factor_indef.cpp,factor_posdef.cpp}`, `include/smf/{threading.hpp,control.hpp,workspace.hpp}`; create `solver/tests/test_task_budget.cpp`; extend parallel/BLAS guard tests.

**Interfaces:** `ExecutionPolicy {int threads; uint64_t memory_budget_bytes; int64_t min_task_flops; bool deterministic;}`. Replace the uncosted traversal with scheduling that consumes predicted/realized front costs and pool reservations. One process/solver thread budget governs task and BLAS parallelism; global thread APIs cannot race across independent instances.

- [ ] Add `TaskBudget.CoarsensTinySubtrees`, `TaskBudget.RespectsLiveMemory`, `TaskBudget.LargeRootMakesProgress`, and `TaskBudget.ConcurrentInstances`. Test deep trees without unsafe recursion growth and cancellation/error propagation without deadlock.
- [ ] Coarsen small subtrees; reuse per-worker arenas; admit ready work only when contribution memory fits. Incorporate realistic pool high-water statistics and conservative delayed-pivot growth handling.
- [ ] Add intra-front tile tasks or an exclusive threaded-BLAS phase for large separators, justified by profiles. Avoid nested oversubscription and process-global BLAS mutation from simultaneous tasks. Test backend selection and actual runtime thread counts.
- [ ] Measure 1/2/4/8/16+ physical cores, throughput mode and declared deterministic mode separately. Compare with the strongest competitor at equal resources, not just SMF's slower serial code. A 4x speedup on 16 cores on parallel-friendly cases is a diagnostic aspiration, not a release claim by itself.
- [ ] Run `TaskBudget|ParallelFactor|ParallelDeterminism|BlasThreadGuard|DelayedPivots`, full numerical regressions, race/deadlock stress and G2/G3 replay. Commit only with memory/time profiles and an explicit threading contract.

## Task 9: Qualify the actual IPOPT integration

**Files:** modify `src/{smf_ma97_plugin.cpp,ipopt_adapter.cpp}`, `include/smf/{ipopt_adapter.hpp,ipopt_mock.hpp}`, `benchmarks/bench_ipopt_compare.cpp`, solver/benchmark/test CMake; create `solver/tests/test_ma97_plugin_abi.cpp`, `solver/tests/test_ipopt_plugin_runtime.cpp`.

**Interfaces:** use the precise installed IPOPT/HSL C interface declarations and versioned compatibility mapping. Keep mocks explicitly test-only. Map supported control fields accurately and reject unsupported requests; report actual ordering/scaling. Plugin runtime tests load the produced shared library rather than a duplicate adapter implementation.

- [ ] Compile ABI checks for `sizeof`, alignment and `offsetof` against authoritative headers, plus exported-symbol signatures. Test native Windows LLP64 separately if support is requested; never assume `long` has Linux widths.
- [ ] Add `Ma97PluginRuntime.ChangedValuesAndInertia`, `Ma97PluginRuntime.MultipleRhsLeadingDimension`, `Ma97PluginRuntime.QualityRetry`, and teardown/error tests. Include exact-Hessian IPOPT cases that exercise negative-eigenvalue checks, not only L-BFGS.
- [ ] Route real repeated calls through the corrected scaling/numeric-workspace API, preserve duplicate-value mappings, and include retry/capture overhead according to benchmark mode. Fix ignored or falsely echoed options.
- [ ] Compare actual IPOPT runs under equal starts, tolerances and supported options; retain convergence failures and report final feasibility/optimality, objective, iterations and total time. A faster factor with worse convergence fails application qualification.
- [ ] Run ABI/runtime tests plus `IpoptAdapter|OcpKktRegression|CApiTest` and the real benchmark. Commit with the exact versions/platforms qualified and explicit unavailable cases.

## Task 10: Earn the first win, then qualify a stronger native portfolio

**Files:** create `solver/include/smf/dispatch_policy.hpp`, `solver/src/dispatch_policy.cpp`, `solver/tests/test_dispatch_policy.cpp`; modify `symbolic_analysis.cpp`, `supernode_detection.cpp`, `factor_posdef.cpp`, `factor_indef.cpp`; extend report tooling and create generated `docs/performance/` reports.

**Interfaces:** `NativePath {TinyDense, BandedSPD, SparseSPD, GeneralIndefinite, CertifiedQuasiDefinite}`; `DispatchDecision select_native_path(const AnalysisKeep&, const Control&, int nrhs, int expected_refactors)` with a recorded reason. Add `Control::expected_refactors=1`. Use structural/numerical invariants and trained cost models, never matrix IDs. Any quasi-definite path verifies its required signed-block assumptions on the actual values and falls back on failure.

- [ ] Review filled-column supernode detection and amalgamation with independent symbolic references. Compare predicted versus actual fill/front shapes; tune amortized analysis+factor cost and memory jointly. Preserve existing narrow-band SPD dispatch instead of reimplementing it.
- [ ] Add tiny dense and certified KKT experiments only where profiles justify them. Validate ordering changes, non-quasi-definite inputs, rank loss and fallback accounting. No static-pivot path for arbitrary KKT matrices.
- [ ] Test dispatch independence from matrix name, bounded setup overhead, valid fallback, and every required numerical case. Fix the policy before opening held-out results; tune only on the training/validation split.
- [ ] Run full G2 first. If it passes, attempt G3. Generate per-instance fastest-baseline ratios, family-weighted geometric means and intervals, losses, memory, coverage and reliability. Include IPOPT total time as a separate outcome; do not substitute linear-solve wins for application wins.
- [ ] Reproduce on a second host before G4. Publish no broad claim with unavailable baselines or failed correctness gates. Commit the policy and reproducible report artifacts; share externally only if separately requested.

## Task 11: Time-box advanced research after the CPU gates

These are ordered experiment plans, not instructions to build all features. Detailed implementation plans follow evidence from the preceding tasks.

| Experiment | Initial time box | Entry requirement | Continue only if |
|---|---|---|---|
| APTP tiled fronts | 2 weeks | Correct blocked reference; pivoting is a measured limiter | >=20% full-sequence gain on the target cohort, same G1 and regression budget, effective robust fallback |
| FP32 factors + FP64 residual/refinement | 1 week | Refinement and inertia qualification exist | >=20% total gain including retries/fallback; no new wrong answers or inertia failures |
| GPU large-front updates | 2 weeks | Recorded fronts, device memory fit, calibrated transfer/compute model predicts a win | >=1.5x full-sequence gain including transfers/setup; accuracy, memory and competitor coverage hold |
| Explicit structured KKT condensation | 2 weeks | Application supplies valid block structure and reconstruction contract | Original KKT residuals, inertia/optimizer semantics and application convergence preserved; >=20% total gain |
| BLR/compression | 2 weeks | Sampled fronts exhibit useful low ranks | Time/memory gain survives original-system accuracy and inertia checks |

- [ ] For each admitted experiment, save the baseline commit, hypothesis, frozen training/holdout subset, proposed file scope, accuracy contract and stop condition before coding.
- [ ] Keep CPU/device results, whole-solver backend results and native-kernel results distinct. Evaluate modern SSIDS/cuDSS versions from their actual documentation; do not assume historical feature restrictions still hold.
- [ ] Disable unsuccessful experiments and record evidence. Two failed performance rounds trigger architecture reassessment, not automatic feature expansion.
- [ ] Plan MA77/out-of-core, distributed memory, complex arithmetic or full 64-bit external indexing only as separate projects with their own resource/competitor contracts. Do not label them supported by this plan's first release.

## Reproducible execution starting point

Commands below target a supported Linux/WSL environment and are **not executed in this planning session**. First inventory dependencies; required competitor access is separate from the core build. Use the repository-standard `build/` location for optimized work and a separate sanitizer directory after compiler-flag cleanup in Task 1.

```bash
cmake -S solver -B build -DCMAKE_BUILD_TYPE=Release \
  -DSMF_BUILD_TESTS=ON -DSMF_BUILD_BENCHMARKS=ON \
  -DSMF_BUILD_C_API=ON -DSMF_BUILD_IPOPT_ADAPTER=ON \
  -DSMF_BUILD_CHOLMOD_COMPARE=ON -DSMF_BUILD_MA97_PLUGIN=ON \
  -DSMF_BUILD_IPOPT_BENCHMARK=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Existing `bench_compare`, `bench_kkt_fixture_compare` and `bench_ipopt_compare` can establish reproduction evidence, but their output is not qualified until Tasks 1/6/9 fix their contracts. Record which conditional CMake targets actually exist; a missing optional target is not a passed test.

First executable slice: **Task 1's numerical qualification tests plus Task 2's two adversarial reproducers**. They determine whether the next hour should repair a wrong answer or optimize a measured bottleneck.
