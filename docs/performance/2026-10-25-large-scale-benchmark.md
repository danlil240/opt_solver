# Large-scale benchmark: SuiteSparse + real IPOPT KKT systems, 1/8/16 threads

**Date:** 2026-10-25
**Purpose:** measure smf where world-class sparse direct solvers compete: n = 10⁴–10⁶, multi-threaded, and the KKT systems an interior-point optimizer actually produces. The earlier optim20 suite (n ≤ 4,000, single-threaded) is where a lean serial code like MA27 looks best.

## Corpus

**SuiteSparse (30 matrices: 18 indefinite, 12 SPD; n = 16,554 – 1,585,478)**, fetched reproducibly by `tools/bench/fetch_suitesparse.py` into `benchmarks/fixtures/large/` (gitignored, ~1.5 GB). The manifest `benchmarks/manifests/large.json` records sha256 hashes.
- Symmetric indefinite: GHS_indef optimization and saddle-point KKT systems (`cvxqp3`, `ncvxqp3/5/9`, `aug3dcqp`, `blockqp1`, `boyd1/2`, `cont-201/300`, `d_pretok`, `turon_m`, `stokes128`, `mario002`), plus other indefinite PDE problems (`helm3d01`, `bratu3d`, `copter2`, `dawson5`).
- SPD: `finan512`, `oilpan`, `thermal1/2`, `cfd2`, `Dubcova3`, `bmwcra_1`, `pwtk`, `nd24k`, `af_shell7`, `apache2`, `G3_circuit`.
- Run "cold" (analyse + factor + solve) on the `recommended` track: pivot threshold u = 0.01, the HSL default.

**IPOPT KKT sequences (7 models, KKT n = 20K–188K)**, captured by `tools/bench/ipopt_kkt.py capture`. IPOPT 3.14 solves Pyomo models using the `smf_ma97` plugin as its linear solver. With `SMF_KKT_DUMP_DIR` set, the plugin records the KKT pattern and the values of every factorization IPOPT requested. Five evenly spaced factorizations (first to last iteration) are kept per model.
- Models: semilinear PDE-constrained optimal control (2 sizes), COPS-style robot arm optimal control, obstacle minimal surface, synthetic AC optimal power flow (2 sizes), factor-model portfolio QP (dense rows).
- Replayed as "warm_sequence" (1 analyse + 5 recorded refactorizations and solves) on the `ipopt` track: pivot threshold 1e-8, which is IPOPT's `ma27_pivtol`/`ma97_u` default (MUMPS gets IPOPT's `mumps_pivtol` 1e-6). Like IPOPT, the harness applies up to 3 steps of iterative refinement around every solve, identically for all solvers, timed in `refine_ms`.

## Solvers

| solver | status |
|---|---|
| smf | this repo (serial lean engine at 1 thread; the old task-parallel path at 8/16) |
| MA27 | CoinHSL archive (serial; run at 1 thread only) |
| MUMPS 5 | conda-forge, sequential MPI, OpenMP/BLAS threads |
| MKL PARDISO | oneMKL 2026.2, threaded (requires the `SMF_MKL_THREADING=intel` build) |
| CHOLMOD | SuiteSparse, SPD only |
| MA57, MA97 | **not measured**: they need the licensed full CoinHSL. Runtime-loaded adapters are ready: set `SMF_HSL_LIBRARY` to the library and they run. MA97's adapter was validated against the MA97-ABI `smf_ma97` plugin. |
| SPRAL SSIDS | **not measured**: no Windows build (no conda-forge package, no Fortran compiler here) |

## Method

- `tools/bench/run.py --threads 1,8,16`: one isolated process per attempt; `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are exported per attempt, and `bench_replay --threads` also sets `mkl_set_num_threads`/`omp_set_num_threads` and smf's `num_threads`. A solver that times out or is unavailable at a thread count is not retried on that case.
- Thread control did not exist before this work. The default build links *sequential* MKL, so earlier "1-thread" numbers were genuinely serial, but nothing could scale. The thread sweep therefore uses a separate build: `-DSMF_MKL_THREADING=intel` → `build-mt/`.
- Accuracy gate unchanged: normwise backward error ≤ 1e-10 and componentwise ≤ 1e-8, on every replayed step.

## Results

### IPOPT end to end (1 thread)

`tools/bench/ipopt_kkt.py e2e`: the same `.nl` model is solved by IPOPT 3.14.20 with each linear solver; times are IPOPT's own wall clock and its `LinearSystemFactorization` timer (results in `benchmarks/results/ipopt_e2e_t1.json` and `ipopt_e2e_noscale_t1.json`). `smf` is the `smf_ma97` plugin with IPOPT's MA97 defaults; `smf_noscale` adds `ma97_scaling none`. MUMPS factorization time is not separated by IPOPT's timers.

| model | vars | MA27 | MUMPS | smf (default) | smf_noscale | MA27 / smf_noscale |
|---|---:|---:|---:|---:|---:|---:|
| pde_control_120 | 28,800 | 0.54 s (fact 0.30) | 1.28 s | 0.85 s (fact 0.48) | 0.70 s (fact 0.34) | 0.78x |
| pde_control_250 | 125,000 | 4.86 s (fact 3.85) | 6.31 s | 32.19 s (fact 30.12) | 4.38 s (fact 2.79) | 1.11x |
| robot_arm_8k | 71,998 | 8.38 s (fact 1.29) | 9.22 s | **timeout (900 s)** | 3.10 s (fact 0.83) | 2.70x |
| obstacle_surface_200 | 40,000 | 11.93 s (fact 3.65) | 19.43 s | 11.86 s (fact 2.98) | 11.01 s (fact 1.80) | 1.08x |
| acopf_grid_80 | 14,931 | 3.88 s (fact 1.96) | 4.11 s | 3.18 s (fact 1.46) | 2.47 s (fact 0.71) | 1.57x |
| acopf_grid_160 | 59,731 | 25.72 s (fact 16.60) | 19.45 s | 22.36 s (fact 15.26) | 13.61 s (fact 5.42) | 1.89x |
| portfolio_20k | 20,040 | 7.65 s (fact 0.13) | 1.52 s | 0.42 s (fact 0.12) | 0.46 s (fact 0.12) | 16.5x |

All runs converge to the same optimum in the same number of iterations (±3 on obstacle_surface).

- **With scaling off, smf is the fastest IPOPT linear solver on 6 of 7 models** (MA27 wins only the smallest, pde_control_120). MA27's losses on robot_arm (4.5 s) and portfolio (7.3 s) are in its symbolic analysis: minimum degree degrades on the portfolio model's dense rows. smf's factorization is faster everywhere, but its **backsolve is ~2x MA27's** (robot_arm 0.44 s vs 0.20 s), which is an open gap.
- **With IPOPT's default MA97 settings, smf is unusable on two models.** A trace of the plugin calls on pde_control_250 shows exactly what happens. The first factorization delays 26,176 of 187,500 pivots. That trips IPOPT's `ma97_switch1 = od_hd_reuse` ("high delay") rule, so the second factorization requests MC64 scaling (`scaling=1`). smf's matching scaling takes **23.8 s** for that single call. On robot_arm it never finishes. The cause is `compute_symmetric_matching` (`solver/src/scaling_matching.cpp`), which is O(n²) or worse. Every augmenting-path search re-fills three length-n arrays, picks the next row by a linear scan over all n rows instead of a heap, and updates duals over all n rows. There is no greedy initial matching either, so every column pays for a full search. MC64 does the same job in roughly O(nnz log n).
- A second plugin gap appears in the same trace. Every later call passes `scaling=0` with the stored scale vector ("reuse"), and MA97 would apply that vector. The plugin ignores it (`ScalingMethod::UserSupplied` is declared but not implemented), so those factorizations are unscaled.
- The 26k delays at u = 1e-8 on the first pde_control_250 iterate are themselves a robustness signal. They come from zero-diagonal constraint rows eliminated before their coupled variables, which is the same pattern as the SuiteSparse KKT failures below.

### SuiteSparse thread sweep (partial)

The 1/8/16-thread sweep was stopped after 17 of 30 matrices (through boyd1, n ≤ 93k) because it monopolised the machine. Its binaries are kept in `build-baseline-2026-10-25/` (gitignored) for before/after runs. The full tables are in `benchmarks/results/large_threads/report_partial.md`. What they showed about smf as it was then:

1. **Indefinite KKT systems break smf and MA27 alike.** Both time out (> 120 s) on cvxqp3, ncvxqp5 and ncvxqp3. smf delays 60k–112k pivots on bratu3d and cont-201, and bratu3d fails the accuracy gate. PARDISO is fast on these but fails the accuracy gate on cvxqp3, stokes128 and ncvxqp3. MUMPS succeeds, slowly, except where its workspace estimate fails (INFOG(1) = −9; the adapter now retries with a larger ICNTL(14)).
2. **SPD analysis, not factorization, was the loss.** On finan512 smf's analyse took 322 ms against a 24 ms factor, while CHOLMOD's analyse took 20 ms. The cause was METIS: `AutoParallel` (the benchmark's SPD setting) always used it above n = 8192, and it was ~85% of analysis.
3. **More threads made smf slower:** 16-thread factor 35 → 303 ms on aug3dcqp, 8 → 65 ms on blockqp1. `num_threads > 1` routed to the old task-parallel engine, which forces serial BLAS and allocates per front.
4. Excluding failures, smf was at 0.92x the best baseline at 1 thread (6 of 11 cases faster), and 0.46x at 8/16 threads.

### Changes made from these results

| change | where |
|---|---|
| MC64-style matching: heap Dijkstra, FIFO for zero reduced cost, MC64's early exit, dense rows/columns set aside in phase 1 | `scaling_matching.cpp` |
| Matching-based compressed ordering (Duff–Pralet): 2×2 pairs from the matching's cycles kept adjacent, AMD on the compressed graph | `ordering_matching.cpp`, `OrderingMethod::Matching` |
| Auto policy for KKT matrices (n ≥ 10,000 with any zero diagonal or ≥ 5% diagonals below 1% of their column): matching computed at analysis; its scaling applied by `ScalingMethod::Auto`, now the default; the matching ordering replaces AMD when its predicted flops ≤ 1.5× AMD's | `symbolic_analysis.cpp`, `factor_indef.cpp` |
| `AutoParallel`: AMD first; METIS only when AMD's predicted flops × `expected_refactors` ≥ 4.5e5 per row | `symbolic_analysis.cpp` |
| `num_threads > 1`: serial engine with BLAS threads in the fronts; old engine opt-in (`Control::tree_parallel`) | `factor_indef.cpp`, `factor_posdef.cpp` |
| Static pivoting, opt-in (`Control::static_pivot_tol`, `Info::static_pivots`): fronts never delay; the benchmark adapter refines after a perturbed factorization | `ldlt_kernel.cpp`, `multifrontal_serial.cpp` |
| `smf_ma97` plugin: IPOPT's reused scale vector applied (`ScalingMethod::UserSupplied`); smf input errors reported as MA97 errors instead of "singular" | `smf_ma97_plugin.cpp` |

### smf after the changes (1 thread, `bench_profile`, ms, analyse + factor + solve)

Measured at below-normal priority on a busy machine (±20%). Thread scaling and the IPOPT runs have not been re-measured yet.

| case | before | after | best baseline (sweep) |
|---|---:|---:|---:|
| cont-201 | 443 | **100** | PARDISO 146 |
| bratu3d | inaccurate | **311** | PARDISO 150 |
| boyd1 | timeout | **172** | MUMPS 692 |
| cont-300 / boyd2 | — | 294 / 216 | — |
| aug3dcqp / stokes128 | 47 / 172 | 66 / 122 | MA27 54 / MA27 168 |
| thermal1 | 278 | **83** | CHOLMOD 81 |
| finan512 | 350 | **67** | CHOLMOD 92 |
| oilpan | 235 | 203 | CHOLMOD 252 |
| cvxqp3 / ncvxqp5 / ncvxqp3 | timeouts | 2.2 s / 10.3 s / 37.5 s with static pivoting (opt-in) | PARDISO inaccurate / 1.25 s / inaccurate; MUMPS — / 4.7 s / 12.4 s |

optim20 against MA27 is unchanged (1.49x geomean). The auto policy starts at n = 10,000, below which the matching's fixed cost outweighed the delays it saves.

- **d_pretok and turon_m are singular**, most likely through the pressure null space typical of these flow problems: with b = 1, smf, PARDISO and MUMPS all reach ω = 1 (smf has the smallest η, 1.7e-16). They do not belong in an accuracy comparison.
- **On the largest SPD matrices smf's factorization is fastest** (G3_circuit 1.4 s vs CHOLMOD 6.6 s, PARDISO 2.5 s), but its METIS analysis is not (5.7–7 s vs 0.75 s and 3.4 s). The new METIS rule picks AMD where that wins a one-off solve: G3_circuit 7.0 s vs 8.2 s, apache2 7.4 s vs 9.6 s.

### Round 2 (2026-10-26): threads and IPOPT

**Threads.** 7 representative matrices, smf rebuilt, baselines from the
earlier run, 3 trials each, `benchmarks/results/threads_v6/report.md`; all 63
smf runs accurate:

| threads | smf faster than every baseline | geomean best-baseline / smf |
|---:|---:|---:|
| 1 | 7 of 7 | 1.27x |
| 8 | 6 of 7 | 1.23x |
| 16 | 6 of 7 | 1.28x |

Before this round smf got *slower* with threads. The remaining loss is bmwcra_1. Its analysis is dominated by serial METIS (~0.6 of 0.85 s), while PARDISO's ordering is parallel.

**IPOPT end to end** (IPOPT's default MA97 options, 1 thread):

| model | MA27 | MUMPS | smf before | smf now |
|---|---:|---:|---:|---:|
| pde_control_120 | 0.54 s | 1.28 s | 0.85 s | **0.53 s** |
| pde_control_250 | 4.86 s | 6.31 s | 32.2 s | **3.29 s** |
| robot_arm_8k | 8.38 s | 9.22 s | timeout | **2.58 s** |
| obstacle_surface_200 | 11.93 s | 19.43 s | 11.86 s | **9.37 s** |
| acopf_grid_80 | 3.88 s | 4.11 s | 3.18 s | **2.19 s** |
| acopf_grid_160 | 25.72 s | 19.45 s | 22.36 s | **11.12 s** |
| portfolio_20k | 7.65 s | 1.52 s | 0.42 s | **0.39 s** |

optim20 against MA27 is unchanged at 1.49x.

**Changes:**
- Tree-parallel factorization for `num_threads > 1` (`multifrontal_serial.cpp`), in three phases:
  - independent subtrees of bounded work on worker threads, with serial BLAS;
  - the top of the tree as a task DAG, under a budget of `num_threads` BLAS threads in total, since oversubscribing MKL's spinning teams was ruinous;
  - a parallel concatenation of the per-thread factors.
  It uses `std::thread`, not OpenMP: MSVC's OpenMP runtime aborts next to the ones MKL and CHOLMOD load. Top-of-tree contribution blocks come from a pool of reusable, uninitialised buffers; fresh zero-filled multi-MB buffers per front cost more than the fronts. `Info::parallel_subtrees` reports when it runs.
- Front kernel: panels update only the pivot-candidate columns, and the trailing Schur complement gets one rank-p update at the end, `L21 D L21ᵀ`, formed in the front's unused upper triangle.
- Column counts by Gilbert–Ng–Peyton (as CSparse's `cs_counts`): nearly O(nnz(A)) instead of O(nnz(L)). Analysis is 1.1–1.8x faster on large matrices.
- `smf_ma97` plugin: the analysis is deferred to the first factorization when IPOPT passes no values, so the KKT matching policy applies inside IPOPT (ACOPF: 117k delays per factorization before). It also sets `expected_refactors = 10`.
- `Solver::factor(ak, span<const double> values, …)`: refactor from values alone, with no `CscLower` copy or pattern comparison. The benchmark adapter no longer copies the matrix; about 290 ms of af_shell7's wall time was the adapter's own copies.

### Still open

- The hardest QP KKT cases (cvxqp3, ncvxqp3) need static pivoting and are still slower than MUMPS. The matching ordering's fill is the cost.
- Serial METIS dominates analysis on bmwcra_1-like matrices. A parallel nested dissection would fix it: order the two halves of the top separator concurrently.
- Chains of thin top fronts, with p of 2–30 pivots on f ≈ 1,800 (helm3d01, copter2), each move a whole contribution block. Fill-tolerant amalgamation, as in CHOLMOD's relaxed supernodes, would merge them.
- ~~The solve phase is serial (PARDISO's scales 2–3x), and inside IPOPT smf's backsolve is still ~1.8x MA27's (robot_arm 0.36 s vs 0.20 s).~~ Done: [2026-10-26-solve-phase.md](2026-10-26-solve-phase.md). The backsolve is at or below MA27's on all 7 IPOPT models (the robot_arm gap was a matching bug), and the solve is tree-parallel (1.4–2.9x at 8 threads).
