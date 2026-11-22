# SMF performance and numerical reliability design

Date: 2026-10-25. Audited revision: `b43b703`.
Status: proposed engineering design; no implementation or new performance measurements in this planning session.
The user delegated workload, hardware, and approach choices. The choices below are recommendations, not measured performance promises.

## Objective and decision

Build a native sparse symmetric solver that wins on real optimization workloads, then earn broader leadership through reproducible comparisons. Start with **repeated real symmetric indefinite KKT systems on multicore CPUs**, retain a competitive SPD path, and introduce accelerator paths only after measured break-even tests.

"Faster than all HSL solvers on every matrix" is not a defensible universal guarantee. HSL contains different algorithms for different mathematical problems. The operational target is **faster than the fastest applicable HSL solver on a frozen workload suite, with at least equal numerical reliability and declared hardware/memory limits**. Broader claims require broader evidence. This does not silently reduce the project to an MA27 comparison.

Keep C++20, lower-triangular CSC, double precision as the reference, and existing useful symbolic infrastructure. Replace deficient numerical and execution components incrementally behind regression gates. Standard BLAS/LAPACK kernels are compatible with a native solver; forwarding the whole problem to another sparse solver is a separately labeled backend result.

## What the repository actually establishes

These are source observations, historical reports, and explicitly labeled deductions. No solver binary was built or benchmarked in this session. CMake was not found on the current PowerShell PATH; the existing build configuration targets a Linux-style toolchain.

| Finding | Evidence at the audited revision | Implication |
|---|---|---|
| Modern HSL comparisons are absent | `solver/benchmarks/CMakeLists.txt:20`; `bench_compare.cpp:978` | Nine synthetic SPD cases and optional MA27/Eigen/CHOLMOD/sequential MUMPS do not establish MA57/86/87/97 performance. |
| Old timings are not release evidence | `docs/benchmark_results.md:40`; `MA97_SOLVER_BREATHING_PLAN.md:2666` | Historical average speed ratios are 0.25 against Eigen and later 0.42 against MA27; the log reports one narrow repeated banded-SPD win. Reproduce before using any number. |
| Incorrect answers can enter speed tables | `bench_compare.cpp:1130`, `:413`, `:1413` | Nonnegative residual is treated as success; repeated solves lack numerical validation; exit status does not enforce correctness. |
| 2x2 D entries are consumed as L entries | `pivoting.cpp:111`; `solve_forward.cpp:137`; `solve_backward.cpp:122` | Repair the factor representation and all solve paths before optimizing them. |
| Secondary pivot test selects the wrong diagonal | `pivoting.cpp:78`; `dense_indef.cpp:151` | A nonsingular front can select a zero 1x1 pivot. |
| Numerical swaps have no corresponding solve permutation | `factor_indef.cpp:403`, `:592`, `:83` | Preserve numeric ordering through assembly, D blocks, and solve metadata. |
| Delayed pivots are not implemented | `factor_indef.cpp:620` | Rejection is counted as a zero eigenvalue. Local rejection does not prove global singularity. |
| Stability tests omit extension rows | `pivoting.cpp:46`, `:69` versus `:94` | Fully summed candidates must be checked against the complete active frontal column. |
| Indefinite updates are scalar | `pivoting.cpp:99`, `:133`; `factor_indef.cpp:652` | Introduce blocked updates; SPD already uses LAPACK/BLAS and is a separate optimization problem. |
| Repeated factorization reconstructs storage | `factor_indef.cpp:389`, `:456`, `:475`, `:62` | Retain numeric workspaces, contribution pools, scatter maps, and reusable execution metadata. |
| Parallelism is only across subtrees | `parallel/task_tree.cpp:37`; `factor_indef.cpp:417` | Large root/separator fronts remain bottlenecks; tiny tasks and memory pressure need scheduling policy. |
| Scaling and matching are incomplete | `solver.cpp:14`; `solve.cpp:11`; `scaling_matching.cpp:21` | Scaling must work with the repeated API; matching currently returns FeatureNotAvailable. |
| Large-storage and integration risks remain | `factor_posdef.hpp:21`; `smf_ma97_plugin.cpp:50`; `ipopt_adapter.hpp:5` | Widen factor offsets; test the real IPOPT plugin and authoritative ABI rather than relying on mock tests. |

All unqualified source paths in this table are beneath `solver/src/`, except named benchmark paths and headers beneath `solver/include/smf/`.

Two exact, source-derived reproducers to execute first:

1. Force a full 2x2 front for `A=[[0,1],[1,0]], b=[1,2]`. The correct answer is `[2,1]` and inertia is `(positive=1, negative=1, zero=0)`. Current forward/D/backward operations algebraically yield `[0,1]` because D's off-diagonal is treated as L. This is a deduction, not an executed regression result.
2. Force a full 3x3 front for `A=[[0.005,1,0],[1,0,2],[0,2,1]]`, `pivot_u=0.01`. The determinant is `-1.02`. The current secondary test selects the zero diagonal in row 1. No nonsingular successful factorization may divide by that zero. Test both sparse and dense helpers.

Retire the old explanation that an enormous residual is automatically acceptable because a matrix is ill-conditioned. Conditioning influences forward error; acceptability requires backward-error analysis of the original system. Preserve old reports as historical artifacts with explicit notices; generate new reports from raw results.

## Approaches considered

| Approach | Benefit | Cost or limit | Decision |
|---|---|---|---|
| Tune current kernels and constants | Short experiments, preserves structure | Cannot repair absent pivot semantics or establish broad reliability | Use only after correctness gates. |
| Dispatch to existing sparse solvers | Fastest route to a stronger application baseline and fallback | Does not establish that native SMF outperforms those solvers | Keep as benchmark adapters/optional product backend, separately labeled. |
| Staged native core with workload selection | Real route to native speed advantages, reuse of existing work | Substantial numerical and performance engineering | Recommended primary approach. |

Do not start with a complete rewrite, machine-learned solver selection, distributed memory, GPU port, or sparse low-rank updates. First collect profiles and prove the CPU implementation. Updating many KKT diagonal entries is not necessarily a low-rank modification.

## Scope and comparison set

Initial supported problem: real symmetric square matrices, SPD or indefinite, single and multiple RHS, including changing values with fixed sparsity. Singular cases have explicit outcomes and form a separate robustness suite. Complex, unsymmetric, rectangular, distributed-memory and out-of-core solving are later independent scopes, not implied capabilities.

| Track | Required comparison set before a claim |
|---|---|
| Repeated/general indefinite, in RAM | Genuine HSL MA27, MA57, MA86, MA97; SPRAL SSIDS; MUMPS; Intel oneMKL PARDISO; Panua PARDISO when accessible |
| SPD | HSL MA87 and MA97, CHOLMOD; applicable MA57/MA86/MUMPS/PARDISO configurations; Eigen as small-case reference |
| External-memory | HSL MA77 plus comparable disk/memory policies; separate extension gate |
| GPU/system-level | Relevant versions of SSIDS and cuDSS plus the strongest CPU baseline; native SMF GPU kernels reported separately from backend dispatch |

MA64 is a dense frontal kernel, not an additional sparse end-to-end competitor. MA77 is a meaningful separate external-memory comparison, not a solver to dismiss for being slower when used in RAM. Treat unavailable/licensed packages as **not measured**. Missing required HSL results block an HSL leadership claim, not independent development.

HSL's own MA97 page recommends MA86 or MA87 for large problems when bit-compatible results are unnecessary. Evaluate reproducible and throughput modes separately. See [HSL MA97](https://www.hsl.rl.ac.uk/catalogue/hsl_ma97.html), [MA86](https://www.hsl.rl.ac.uk/catalogue/hsl_ma86.html), [MA87](https://www.hsl.rl.ac.uk/catalogue/hsl_ma87.html), [MA77](https://www.hsl.rl.ac.uk/catalogue/hsl_ma77.html), and [MA64](https://www.hsl.rl.ac.uk/catalogue/hsl_ma64.html).

## Numerical contract

- For nonsingular success, every answer is finite and solves the **original** unscaled, unregularized system to normwise backward error `eta_inf <= 1e-10` and componentwise backward error `omega <= 1e-8`. These are initial qualification gates, not universal accuracy theorems. Any future class-specific exception must be declared before timing and shown separately.
- `eta_inf = ||b-Ax||inf / (||A||inf*||x||inf + ||b||inf)`. `omega = max_i |b-Ax|_i / (|A||x|+|b|)_i`; a zero denominator gives zero only when its residual is zero, otherwise infinity. Validate every RHS and every numeric update, with safe/compensated accumulation; use an independent implementation for validation.
- Record relative residual and forward error too. Require relative forward error `<=1e-8` for well-conditioned manufactured systems with known solutions. Do not demand small forward error on ill-conditioned systems without a conditioning argument.
- Inertia must match a dense spectral/reference oracle on small, well-separated spectra and mathematically constructed inertia fixtures. Near-zero classification uses one documented, scale-aware policy; ambiguous near-zero spectra are reported rather than compared to a falsely exact oracle.
- Factor identity uses `P^T S A S P = L D L^T`, with `perm[new]=old`, P including symbolic and numeric permutations, S positive diagonal, and 1x1/2x2 D stored distinctly from unit-block L. Test the identity independently of solve.
- No silent regularization or falsely successful singular solve. Optimizer-requested shifts are explicit inputs. Any optional perturbation reports the changed system and also validates the original residual.
- A rejected nonroot pivot becomes a delayed column carried to its parent. Root rank decisions follow a documented numerical threshold. Delays, zero pivots, and rank are separate statistics.
- Scaling, refinement, retries, fallback and transfers count toward the relevant solve/sequence time. FP64 factors are the reference path; faster precision modes require the same external accuracy gate.

The distinction between D storage, permutations and L is grounded in [LAPACK DSYTF2](https://netlib.org/lapack/explore-html/db/d8d/group__hetf2_ga91a9fdb3b98a38e1a91689b65c366569.html). For sparse front stability, implement a documented threshold strategy; dense Bunch-Kaufman on the pivot block alone is insufficient.

## Target architecture

1. **Immutable symbolic plan:** canonical structure, original-entry/duplicate maps, ordering, elimination/front tree, scatter/assembly maps, predicted work and memory. Analysis must not depend on later mutation of `AnalysisKeep::cleaned.values`.
2. **Reusable numeric workspace:** scaled/permuted values, per-worker scratch, contribution pool, dynamic delayed-column metadata, and capacity-managed factor buffers. Recompute value-dependent scaling and validate any reused pivot decisions. Capacity growth is allowed and measured; no zero-allocation claim for arbitrary pivot changes.
3. **Owned completed factors:** explicit L, D, total permutation, realized front shapes and scaling. Factors remain valid after symbolic input lifetime ends. 64-bit storage offsets and checked products precede large-fill claims.
4. **Caller-owned solve workspace:** reentrant independent solves; batched RHS via TRSM/GEMM above a calibrated crossover; tiny-front kernels below it. Preserve existing SPD/banded implementations and test dispatch cost.
5. **Reliable reference indefinite engine:** correct serial threshold pivoting, swaps, safe 2x2 arithmetic, delay propagation, inertia, scaling, refinement and errors. Share numerical logic between serial and parallel execution.
6. **Fast indefinite engine:** blocked panels and BLAS-3 Schur updates, then a separately qualified a posteriori threshold pivoting (APTP) experiment. APTP is supported by published SSIDS work; it is not an unvalidated license to remove pivot checks. See [Duff, Hogg and Lopez, STFC report](https://epubs.stfc.ac.uk/work/40018255).
7. **Costed CPU execution:** coarsen tiny subtrees, bound live contribution memory, schedule independent large-front tiles, respect NUMA placement and one total CPU budget. Avoid concurrently changing process-global BLAS thread settings. Keep deterministic mode explicit.
8. **Conservative selection:** use bandwidth, front widths, predicted fill, repeat count, RHS count and measured work, never matrix names. Select native SPD/banded, tiny dense, general indefinite, or explicitly certified quasi-definite KKT paths. Unsupported structure falls back to the robust native engine, with fallback time included.

Matching must preserve the symmetric problem and integrate with ordering, including paired pivots where appropriate. A generic unsymmetric row permutation must not be applied to a symmetric LDLT path. Measure its setup/fill cost and actual reduction in delays; do not hard-code it on every problem.

## Benchmark contract and claim gates

Freeze manifests with SHA-256, provenance, symmetry/type, dimensions, memory limits, timeout, sequence IDs, expected outcomes and train/validation/test assignment. Initial full corpus: at least 40 SPD matrices, 60 nonsingular indefinite matrices, 20 separate adversarial/singular fixtures, and 30 KKT sequences across at least 10 independent application families. Split KKT families 6/2/2 for tuning/validation/held-out testing; split other matrices by originating family, not random variants of the same matrix. Keep public manufactured smoke fixtures small.

Use real [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) matrices and real optimization sequences, plus generated controlled fixtures. Capture changing values and RHS, early/late iterations, barrier changes, inertia corrections and failed attempts. Retain failures; an artificial SPD objective Hessian is not a substitute for the constrained indefinite KKT matrix.

Measure separately:

- Cold public API: conversion, ordering, analysis, factor, solve, refinement and required setup.
- Warm sequence: analysis once, then **changed** numeric values, each factor and all RHS, including retries and workspace growth.
- Existing-factor RHS throughput, with RHS counts 1/4/16/64.
- Real IPOPT total time, success, final feasibility/optimality, objective, iterations, factorizations and retries under identical starts and stopping criteria.

Primary qualification machine: a dedicated Linux x86-64 host with at least 16 physical cores and 64 GiB RAM, fully recorded. Broader claims require another CPU architecture/vendor and a larger-memory host. These are desired resource specifications, not claims about hardware currently available. Use available hardware for development; rent/buy nothing without separate user authorization.

Run 1/2/4/8/16 and available higher physical-core counts, pin affinity, record NUMA, compiler, flags, BLAS, integer ABI, library versions and actual runtime threading. Common resources for every solver. Separate warmup from at least 10 measured trials, randomize solver order, preserve raw samples, median/IQR and family-bootstrap 95% confidence intervals. For tiny cases batch enough operations to measure reliably. Performance runs use isolated processes; sanitizer runs are correctness-only.

Have two tracks: shared permutation/scaling where the APIs support it for diagnosis, and each solver's documented recommended/tuned configuration with a fixed tuning budget for headline results. Unsupported common-ordering configurations are reported as such. Time adaptation costs; do not tune on held-out matrices.

Report wall time, peak resident memory, factor storage, allocation/growth counts, fill/flops, pivot/delay counts, refinement, failures, and performance profiles. Virtual address peak is not resident memory. Use [Dolan-More performance profiles](https://doi.org/10.1007/s101070100263); failed, inaccurate, OOM and timed-out attempts never count as fast solves.

For each instance, the speed comparator is the minimum valid time among the declared eligible baselines, not an arithmetic average of weak competitors. Native SMF failures block the target gate; baseline failures remain in reliability tables/profiles. Calculate speed geometric means on jointly valid instances and disclose coverage; baseline failures do not create infinite speedup credit. Weight application families equally and bootstrap whole families/sequences rather than treating correlated iterations as independent samples.

| Gate | Requirement | Meaning |
|---|---|---|
| G0: evidence | Strict harness, actual backend identity, preserved failures, fixed corpus | Timings become interpretable. |
| G1: correctness | All required numerical/ABI tests pass; no false success; stated accuracy and inertia | Performance optimization can be qualified. |
| G2: first KKT win | Held-out sequence geometric mean speedup >=1.2 vs fastest eligible baseline; lower 95% CI >1.0; >=60% sequence wins; no new failure | Minimum credible first win. |
| G3: strong KKT lead | Geometric mean >=1.5 (2.0 stretch), >=70% wins, p95 time ratio <=1.25, peak RSS <=1.25 of fastest valid HSL configuration on each jointly solved case | Useful dominance within the declared workload. |
| G4: broader lead | Geometric mean >=1.2 and >=60% wins separately on SPD and general indefinite held-out cohorts, on two CPU platforms; no new failures; same regression/memory budget | A broader claim, still explicitly scoped. |

For G2 use the same p95 time-ratio and RSS budgets as G3. A win requires a median time advantage of at least 5%; closer values are ties. Publish every loss. Missing required baselines or inadequate jointly valid coverage prevents the relevant claim; require at least 90% jointly valid coverage within each headline cohort. Numerical qualification applies regardless of timing coverage.

## Delivery, effort, and stop rules

Planning allowances assume one numerical specialist, one C++/performance engineer and one benchmark/integration engineer, with stable hardware and baseline access. These are estimates, not deadlines or assured speedups.

| Window | Deliverable | Release decision |
|---|---|---|
| First 2-3 working days | Reproduce adversarial cases; fix measurement qualification; inventory real baselines; capture first representative KKT sequence | A truthful baseline and ranked profiles, not a speed claim. |
| Weeks 1-4 | Correct pivot representation, delayed pivots, stable 2x2 arithmetic, error paths, real integration validation | G0/G1; extend if correctness is unresolved. |
| Weeks 4-8 | Reusable numeric state, scaling/refinement, blocked updates, calibrated tiny paths | Attempt G2; profile where time remains. |
| Weeks 8-12 | Matching/order selection, memory-aware CPU scheduling, actual IPOPT qualification | Attempt G3; no guaranteed achievement. |
| Months 3-6+ | Broader fronts/corpus/hardware, APTP and conditional GPU/precision studies | Attempt G4 only after evidence. |

A broad competitive native solver may require 6-12+ months or more. Agents can accelerate independent work and review; they cannot replace numerical validation or guarantee a research result.

At every phase, save a reference commit, raw benchmark artifacts and a bottleneck profile. Stop an experiment after its allotted window if it cannot improve the primary metric without violating G1. Reassess the architecture when two successive performance experiments miss their gate. Do not keep adding speculative features to avoid a failed comparison.

GPU entry requires measured large-front work, memory fit and a positive full-sequence transfer-inclusive break-even estimate. Mixed precision requires residual/refinement convergence and correct inertia; fall back to FP64 and include all fallback cost. BLR/compression requires sufficiently low observed ranks and application accuracy/inertia qualification. All are off by default until held-out validation. External-memory leadership needs its own implementation and MA77 tests.

## Builder discretion and open dependencies

The implementer may choose internal containers, panel sizes, pool implementation and task cutoffs using training profiles. No permission is needed for routine code decisions within these contracts. Exact hardware availability, licensed baseline access and real application fixtures are dependencies to inventory in Task 1, not reasons to stop this plan. Their absence narrows which performance claims can be made.

Implementation sequence and testable work packages: [performance implementation plan](../plans/2026-10-25-solver-performance.md).
