# smf IPM mode design

Date: 2026-10-26. Branch: `feat/solver-performance` (head `a7f94ad` plus uncommitted solve-phase work).
Status: approved design, not yet implemented. Horizon 1, step 2 of `docs/roadmap.md`.

## Objective

Make smf the best linear solver for interior-point methods, not only a fast general sparse LDLᵀ. Three features were requested:
1. **Pivot-sequence reuse** across IPM iterations.
2. **Cheap inertia-correction refactors.**
3. **A static/quasidefinite path, as the default on the IPOPT track.**

They are built behind a new engine-level `smf::IpmSession`, which the IPOPT MA97 plugin uses. They are judged on a new corpus of hard (nonconvex, degenerate) NLPs, because the current corpus barely exercises them.

## Evidence that shaped the design

Measured 2026-10-26, 1 thread: `bench_replay` on the recorded IPOPT KKT sequences (5 factorizations each, `ipopt` track, u = 1e-8), plus the IPOPT logs in `benchmarks/results/ipopt_work/e2e/`.

| model | n | delays (5 factorizations) | inertia corrections in IPOPT | replay outcome |
|---|---:|---:|---|---|
| pde_control_250 | 187,500 | 0 | 0 (12 factorizations, 12 iterations) | Success |
| robot_arm_8k | 119,998 | 41,603 | 3 of 31 iterations regularized | Success |
| acopf_grid_160 | 110,931 | 3,994 | 0 | **Inaccurate** (fails the accuracy gate) |
| acopf_grid_80 | 27,731 | 969 | 0 | Success |
| obstacle_surface_200 | 40,000 | 0 | 1 of 133 | Success |
| portfolio_20k | 20,081 | 0 | 0 | Success |
| pde_control_120 | 43,200 | 0 | 0 | Success |

What follows from this:
- **Inertia correction is almost never triggered on this corpus,** because the models are close to convex. Faster correction refactors cannot be shown here.
- **Delays matter only on robot_arm,** and its factorization is already cheap: 75 ms for 5 factorizations.
- **Static/quasidefinite mode is not a speedup on this corpus.** Its value is robustness and a fixed symbolic structure for the planned GPU path (roadmap Horizon 2).

Therefore Phase 0 builds a corpus that does exercise these features before any of them are built.

## Phase 0 — Hard corpus and instrumentation

**Models.** All are built in Pyomo and written as AMPL `.nl` files, the same route as `tools/bench/ipopt_kkt.py`:
- **PGLib-OPF AC-OPF through Egret:** about 8 cases from ~100 to ~10k buses, including the harder `api` and `sad` variants.
  - A new `tools/bench/fetch_pglib.py` downloads the case files and pins each one by sha256, as `fetch_suitesparse.py` does.
  - Fixtures go under `benchmarks/fixtures/` (gitignored, like the SuiteSparse set).
- **COPS problems ported to Pyomo:** glider, rocket, catmix, hanging chain, pinene, steering, elec, polygon.
  - elec and polygon are strongly nonconvex and are expected to trigger inertia corrections.
  - These are new model builders in `tools/bench/ipopt_kkt.py` (or a sibling module if that file grows too large).

**Split.** Each model is tagged `split: "tuning"` or `split: "heldout"` in a new manifest, `benchmarks/manifests/ipopt_hard.json`.
- At least half of the PGLib cases and at least 2 COPS problems are held out.
- Held-out models are used only for the final gates in this spec, never for choosing thresholds.

**Instrumentation.**
- **Plugin trace** (`SMF_PLUGIN_STATS`/trace): per factorization, records whether it is an inertia-correction refactor (a factorization followed by another factorization with no solve in between), plus delays, static pivots, the inertia (pos/neg/zero) and the time.
- **`ipopt_kkt.py e2e`:** adds per model the number of factorizations, inertia corrections, total delays, factorization time and the failure status.
- **`ipopt_kkt.py capture`:** gains an option to keep every factorization of a sequence, instead of 5 evenly spaced ones, so correction sequences can be replayed.

**Baseline.** MA27, MUMPS and current smf on every model, 1 thread, saved as `benchmarks/results/ipopt_hard_baseline_t1.json`.

**Triage.** Find out why acopf_grid_160 fails the replay accuracy gate (backward error ≤ 1e-10 after up to 3 refinement steps, u = 1e-8). Fix it if the cause is a bug; otherwise record the cause in the write-up.

**Exit criterion.** At least 3 tuning models where at least 10% of factorizations are inertia corrections, **or** where total delays exceed 1% of n per factorization. If the corpus does not meet this, add models before Phase 1.

## Architecture

### `IpmSession` (new: `solver/include/smf/ipm_session.hpp`, `solver/src/ipm_session.cpp`)

```cpp
namespace smf {

enum class StaticMode { Off, Fallback, Regularized };

struct IpmControl {
  bool       learn_inertia          = true;  // infer the expected #neg from accepted factors
  bool       abort_on_wrong_inertia = true;  // stop once #neg exceeds the expectation
  double     reanalyse_delay_frac   = 0.01;  // delays > frac * n -> try realized-order re-analysis
  double     reanalyse_flops_ratio  = 1.0;   // adopt only if predicted flops <= ratio * last actual
  int        reanalyse_min_gap      = 5;     // factorizations between re-analysis attempts
  StaticMode static_mode            = StaticMode::Off;
  double     reg_rho                = 1e-8;  // Regularized: rho relative to max|A| after scaling
};

enum class IpmFactorStatus { Ok, WrongInertia, Singular, Error };

struct IpmStats {
  int factorizations = 0, accepted = 0, aborted = 0;
  int reanalyses_tried = 0, reanalyses_adopted = 0, reanalyses_reverted = 0;
  long static_pivots = 0;
  double flops_avoided = 0.0;  // predicted flops of the fronts skipped by aborts
};

class IpmSession {
 public:
  ErrorCode analyse(const CscLower& pattern, std::span<const double> values,
                    const Control& ctrl, const IpmControl& ipm);
  IpmFactorStatus factor(std::span<const double> values,
                         std::optional<int> expected_negatives = {});
  int solve(double* b, int n, int nrhs);  // a solve marks the last factor "accepted"
  InertiaCounts inertia() const;  // existing type, smf/dense_indef.hpp
  const Info& info() const;
  const IpmStats& stats() const;
};

} // namespace smf
```

- **Ownership:** the session owns the `Solver`, the current `AnalysisKeep` (plus the previous one while a re-analysis is on probation), the `FactorKeep`, the workspaces and the history: the last accepted inertia and per-factorization delays/flops.
- **Expected negatives:** a caller that knows the expected negative count passes it explicitly; that would be a future smf barrier solver or a native IPOPT interface. Otherwise the session learns it (Section "Inertia learning").
- **Safety:** if `solve` is called after `factor` stopped early, the session first completes a full factorization of the same values, then solves. A wrong expectation costs time, never correctness.

### Engine hooks

1. **`Control::max_negative_pivots`** (int, −1 = off).
   - Serial engine (`multifrontal_serial.cpp`): after each front, if the running negative count exceeds the limit, return a new `FactorStatus::InertiaExceeded`.
   - Tree-parallel engine: subtrees add to a shared atomic negative count and poll a shared stop flag between fronts.
   - `Info` reports the partial negative count and the predicted flops of the skipped fronts.
2. **Analysis from a given order with 2×2 pairs.** This implements `OrderingMethod::UserSupplied`; the `Solver::analyse` comment currently says it returns `FeatureNotAvailable`.
   - New `Control` inputs: `const Int* user_perm` (perm[new] = old) and an optional list of index pairs to keep adjacent.
   - Supernode detection and amalgamation never split a pair. This reuses the compressed-graph handling in `ordering_matching.cpp` where it fits.
3. **`realized_pivot_order(const AnalysisKeep&, const FactorKeep&)`.** Returns the actual elimination order and its 2×2 pairs: fronts in postorder; within each front, its accepted pivots in pivot order, including columns delayed into it from children. Columns that ended as zero pivots at a root come last.

### Plugin (`smf_ma97_plugin.cpp`)

- `ma97_analyse_d` stores the pattern (analysis stays deferred to the first factorization, which has values).
- `ma97_factor_d` calls `IpmSession::factor`; `WrongInertia` is reported as flag 0 with `num_neg` = the partial count.
- `ma97_solve_d` calls `IpmSession::solve`.
- Scaling options, KKT dumps and statistics stay in the plugin.
- The IPM options are read from environment variables (`SMF_IPM_ABORT`, `SMF_IPM_REANALYSE`, `SMF_IPM_STATIC=off|fallback|regularized`) until the default decision below is made. The MA97 interface has no slots for them.

## Feature 1 — Inertia learning and early abort

- **What IPOPT does:** after each factorization IPOPT compares the reported negative count with the number of constraints m. If they differ, it refactors with a larger δ_w (and, when the matrix is singular, a δ_c > 0). The pattern does not change and neither does m, so the expectation is constant per analysis.
- **Learning:** `expected_neg` is the negative count of the most recent factorization that was followed by a solve. It is unset until the first accepted factorization and reset by a new analysis.
- **Abort:** during factorization the negative count only grows. When it exceeds `expected_neg`, the inertia is certainly wrong, and the factorization stops with `InertiaExceeded`.
  - Too *few* negatives can only be known at the end. Those factorizations run in full, unchanged.
- **Invariant, which is also the test:** an abort never changes a factorization that IPOPT accepts. IPOPT's iteration log must be identical with and without the abort: iterations, objective and the δ_w sequence. Only the time changes.
- **Gate:**
  - Identical logs on every corpus model.
  - Lower IPOPT factorization time on tuning models that have inertia corrections.
  - If the saving is under 5% on every model, the feature ships off by default.
- **No other saving is available in a δ refactor:** the analysis is already reused, IPOPT passes its scaling back in, and the value permutation is O(nnz).

## Feature 2 — Pivot-sequence reuse through re-analysis from the realized order

**Approach.** Re-analyse from the realized pivot order. Rejected alternative: replaying the pivot sequence without threshold checks (MA57's fast factorization). Interior-point values swing too much over a run (Σ goes to 0 or ∞), and the threshold test is cheap.

- **Trigger:** an accepted factorization with delays > `reanalyse_delay_frac` · n, at least `reanalyse_min_gap` factorizations after the last attempt.
- **Re-analysis:**
  - take `realized_pivot_order` of that factorization;
  - analyse with that order and its pairs (hook 2);
  - each previously delayed column now starts in the front where it was actually eliminated, so at the next factorization it is an ordinary pivot there. Threshold pivoting stays on, so new delays are still handled dynamically.
- **Adoption:** only if the new predicted flops ≤ `reanalyse_flops_ratio` × the last factorization's actual flops, which already include the cost of the delays.
- **Probation:** the previous analysis is kept for one factorization. If that factorization delays more pivots than the last one under the old analysis, the session reverts and stops re-analysing this pattern.
- **Side effect:** a different order changes rounding, so IPOPT's iteration path may shift. This feature is gated on outcomes, not on identical logs.
- **Gate:** on tuning models whose delays exceed 1% of n:
  - total factorization time including re-analysis improves;
  - IPOPT iterations stay within ±5%;
  - no new failures.

## Feature 3 — Static/quasidefinite mode and the default decision

**Modes** (`IpmControl::static_mode`):
- **`Off`:** threshold pivoting with delays (today).
- **`Fallback`:** threshold pivoting, but a column that would reach a root unpivoted, or a front whose delays exceed a budget (default: its own pivot count), gets a static pivot (the existing `force_static_pivot`) instead.
- **`Regularized`** (the quasidefinite path):
  - **Regularize:** add +ρ to the primal diagonal and −ρ to the constraint diagonal, with ρ = `reg_rho` × max|A| after scaling. Constraint rows are those with a zero or negative structural diagonal (the KKT auto policy in `symbolic_analysis.cpp` already detects zero diagonals).
  - **Factor:** in a fixed order (AMD; no matching ordering), with 1×1 pivots, no pivot search and no delays. A pivot with |d| below ρ is raised to ρ and counted as a static pivot.
  - **Solve:** `IpmSession::solve` refines against the unregularized matrix, up to `Control::refinement_max_iters` steps.
  - **Why this matters later:** the symbolic structure is fixed at analysis, which the GPU path needs.

**Inertia accounting.**
- With −ρ on the constraint block, the regularized matrix has exactly m negative eigenvalues if and only if H + JᵀJ/ρ is positive definite. For small ρ that is IPOPT's condition: H positive definite on the null space of J. So the inertia check remains meaningful.
- Each pivot raised by static pivoting (in `Fallback` or `Regularized`) is reported as a **zero** eigenvalue, never as a guessed sign. IPOPT treats that as a singular matrix and adds its own δ_c, so its own safeguard takes over instead of smf returning an uncertain inertia.

**Default decision.** The requested goal is "static mode on by default on the IPOPT track". It is decided by measurement on the **held-out** models: `Fallback` and `Regularized` are each compared with `Off`. A mode becomes the plugin default only if:
- it has no more IPOPT failures than `Off`;
- IPOPT iterations stay within ±5% (geometric mean);
- IPOPT wall time is equal or better (geometric mean);
- every replayed solve meets the accuracy gate (backward error ≤ 1e-10).

If both pass, `Regularized` wins, because it also unblocks the GPU path. If neither passes, `Off` stays the default and the write-up records why.

## Testing

All suites are gtest. Each new suite gets its own `add_test` line in `solver/tests/CMakeLists.txt`; ctest runs only listed suites.

- **Engine hooks (`test_ipm_hooks.cpp`):**
  - `max_negative_pivots`: aborts with `InertiaExceeded` when the limit is exceeded; no abort at exactly the limit; serial and tree-parallel engines.
  - `UserSupplied` order with pairs: each pair ends up in one supernode, and the factorization matches the AMD-ordered one to rounding.
  - Round trip: factor a matrix that delays, take `realized_pivot_order`, re-analyse, factor the same values again with 0 delays.
- **Regularized mode (`test_ipm_regularized.cpp`):**
  - a constructed KKT with indefinite H that is positive definite on null(J) gives exactly m negatives;
  - one that is not positive definite on null(J) gives more than m;
  - a forced tiny pivot is reported as zero;
  - the refined solve reaches backward error ≤ 1e-10 against the unregularized matrix.
- **Session (`test_ipm_session.cpp`)**, on the small replay fixtures (pde_control_120, acopf_grid_80 and the new COPS sequences):
  - learns the expected count;
  - aborts on wrong inertia;
  - completes the factorization when a solve follows an abort;
  - re-analysis probation reverts when delays grow.
- **End to end:** `ipopt_kkt.py e2e` on the hard corpus for each feature's gate, including the identical-log check for Feature 1.

## Rollout

Each step is its own commit, with ctest green:
1. Phase 0: corpus, instrumentation, baseline, acopf_grid_160 triage.
2. Engine hooks 1–3 with their tests.
3. `IpmSession` with every feature off, and the plugin moved onto it. **No behaviour change:** IPOPT logs identical to the baseline.
4. Feature 1, early abort.
5. Feature 2, realized-order re-analysis.
6. Feature 3, static modes.
7. Default decision on the held-out set; plugin defaults updated.
8. Write-up `docs/performance/YYYY-MM-DD-ipm-mode.md` with tables for every gate.

**Machine:** the development PC is also used for gaming. IPOPT runs are single-threaded at below-normal priority, and any run expected to take longer than ~10 minutes is announced before it starts.

## Out of scope

- A native IPOPT linear-solver interface (it would receive the expected negative count directly). It can be added on top of `IpmSession` later.
- GPU factorization (roadmap Horizon 2).
- MA57/MA97 baselines (waiting for the HSL academic licence), CUTEst (needs a Linux host), and SSIDS.
- Thread-sweep benchmarking. The parallel abort is unit-tested, not benchmarked.
